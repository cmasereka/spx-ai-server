"""
Auth routes: login, refresh, validate-invite, register.
No authentication required for any of these endpoints.
"""

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from api.auth import (
    create_access_token, create_refresh_token, decode_token,
    hash_password, verify_password, get_db,
)
from src.database.models import User, Invitation

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    code: str
    email: str
    full_name: str
    password: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if user.status == "pending_approval":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending admin approval",
        )
    if user.status == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is suspended",
        )

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    uid = str(user.id)
    return LoginResponse(
        access_token=create_access_token(uid, user.role),
        refresh_token=create_refresh_token(uid),
        user={
            "id": uid,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "status": user.status,
        },
    )


@router.post("/refresh", response_model=RefreshResponse)
def refresh_token(payload: RefreshRequest, db: Session = Depends(get_db)):
    data = decode_token(payload.refresh_token)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    user_id = data.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.status == "suspended":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or suspended")
    return RefreshResponse(access_token=create_access_token(str(user.id), user.role))


@router.get("/validate-invite")
def validate_invite(code: str, db: Session = Depends(get_db)):
    inv = db.query(Invitation).filter(Invitation.code == code).first()
    if not inv:
        return {"valid": False, "note": None, "reason": "Invite code not found"}
    if inv.is_used:
        return {"valid": False, "note": inv.note, "reason": "Invite code already used"}
    if inv.cancelled_at is not None:
        return {"valid": False, "note": inv.note, "reason": "Invite code has been cancelled"}
    if inv.expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
        return {"valid": False, "note": inv.note, "reason": "Invite code expired"}
    return {"valid": True, "note": inv.note, "invited_email": inv.invited_email}


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    # Validate invite
    inv = db.query(Invitation).filter(Invitation.code == payload.code).first()
    if not inv or inv.is_used:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or already-used invite code")
    if inv.cancelled_at is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invite code has been cancelled")
    if inv.expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invite code has expired")

    email = payload.email.lower().strip()

    # Enforce email match if the invitation was issued to a specific address
    if inv.invited_email and inv.invited_email != email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This invite was issued to a different email address",
        )

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(
        email=email,
        full_name=payload.full_name.strip(),
        hashed_password=hash_password(payload.password),
        role="user",
        status="pending_approval",
        invited_by=inv.created_by,
    )
    db.add(user)

    inv.is_used = True
    inv.used_by = user.id
    inv.used_at = datetime.now(timezone.utc).replace(tzinfo=None)

    db.commit()
    return {"message": "Account created — awaiting admin approval", "email": email}
