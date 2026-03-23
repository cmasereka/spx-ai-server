"""
User self-service routes: profile, broker configs.
All endpoints require an approved account.
"""

import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from api.auth import get_approved_user, get_db, hash_password, verify_password, encrypt_credentials
from src.database.models import User, UserBrokerConfig

router = APIRouter(prefix="/api/v1/me", tags=["User"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ProfileResponse(BaseModel):
    id: str
    email: str
    full_name: str
    phone: Optional[str]
    role: str
    status: str
    created_at: Optional[str]
    last_login_at: Optional[str]


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


class BrokerConfigRequest(BaseModel):
    broker_type: str = "tastytrade"
    label: Optional[str] = None
    account_number: str
    provider_secret: str
    refresh_token: str


class BrokerConfigResponse(BaseModel):
    id: str
    broker_type: str
    label: Optional[str]
    account_number: str
    status: str
    created_at: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=ProfileResponse)
def get_profile(user: User = Depends(get_approved_user)):
    return ProfileResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        role=user.role,
        status=user.status,
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


@router.patch("")
def update_profile(
    payload: UpdateProfileRequest,
    user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    if payload.full_name:
        user.full_name = payload.full_name.strip()

    if payload.phone is not None:
        user.phone = payload.phone.strip() or None

    if payload.new_password:
        if not payload.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="current_password is required to change password",
            )
        if not verify_password(payload.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        user.hashed_password = hash_password(payload.new_password)

    db.merge(user)
    db.commit()
    return {"message": "Profile updated"}


@router.get("/broker-configs", response_model=list[BrokerConfigResponse])
def list_broker_configs(
    user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    configs = db.query(UserBrokerConfig).filter(UserBrokerConfig.user_id == user.id).all()
    return [
        BrokerConfigResponse(
            id=str(c.id),
            broker_type=c.broker_type,
            label=c.label,
            account_number=c.account_number,
            status=c.status,
            created_at=c.created_at.isoformat() if c.created_at else None,
        )
        for c in configs
    ]


@router.post("/broker-configs", status_code=status.HTTP_201_CREATED)
def add_broker_config(
    payload: BrokerConfigRequest,
    user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    encrypted = encrypt_credentials({
        "provider_secret": payload.provider_secret,
        "refresh_token": payload.refresh_token,
    })
    cfg = UserBrokerConfig(
        user_id=user.id,
        broker_type=payload.broker_type,
        label=payload.label,
        account_number=payload.account_number,
        encrypted_credentials=encrypted,
        status="pending_approval",
    )
    db.add(cfg)
    db.commit()
    return {"message": "Broker config submitted for admin approval", "id": str(cfg.id)}


@router.delete("/broker-configs/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_broker_config(
    config_id: str,
    user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    cfg = db.query(UserBrokerConfig).filter(
        UserBrokerConfig.id == config_id,
        UserBrokerConfig.user_id == user.id,
    ).first()
    if not cfg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Broker config not found")
    db.delete(cfg)
    db.commit()
