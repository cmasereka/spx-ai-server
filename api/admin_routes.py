"""
Admin routes: user management, invitations, broker config approvals.
All endpoints require admin role.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import get_admin_user, get_db
from src.database.models import User, Invitation, UserBrokerConfig

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateInviteRequest(BaseModel):
    note: Optional[str] = None
    email: Optional[str] = None


class UpdateInviteRequest(BaseModel):
    note: Optional[str] = None
    email: Optional[str] = None


class UpdateUserStatusRequest(BaseModel):
    status: str  # 'approved' | 'suspended'


class UpdateBrokerConfigStatusRequest(BaseModel):
    status: str  # 'approved' | 'rejected'


class UpdateBrokerConfigRequest(BaseModel):
    label: Optional[str] = None
    account_number: Optional[str] = None
    broker_type: Optional[str] = None
    is_paper: Optional[bool] = None


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------

@router.post("/invitations", status_code=status.HTTP_201_CREATED)
def create_invitation(
    payload: CreateInviteRequest,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    code = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7)
    inv = Invitation(
        code=code,
        created_by=admin.id,
        note=payload.note,
        invited_email=payload.email.lower().strip() if payload.email else None,
        expires_at=expires_at,
    )
    db.add(inv)
    db.commit()
    return {
        "id": str(inv.id),
        "code": code,
        "invite_url": f"{FRONTEND_URL}/register?code={code}",
        "expires_at": expires_at.isoformat(),
        "note": payload.note,
        "invited_email": inv.invited_email,
    }


@router.get("/invitations")
def list_invitations(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Invitation, User)
        .outerjoin(User, Invitation.created_by == User.id)
        .order_by(Invitation.created_at.desc())
        .all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return [
        {
            "id": str(i.id),
            "code": i.code,
            "invite_url": f"{FRONTEND_URL}/register?code={i.code}",
            "note": i.note,
            "invited_email": i.invited_email,
            "is_used": i.is_used,
            "is_cancelled": i.cancelled_at is not None,
            "used_at": i.used_at.isoformat() if i.used_at else None,
            "cancelled_at": i.cancelled_at.isoformat() if i.cancelled_at else None,
            "expires_at": i.expires_at.isoformat(),
            "is_expired": i.expires_at < now and not i.is_used and i.cancelled_at is None,
            "created_at": i.created_at.isoformat() if i.created_at else None,
            "created_by_name": u.full_name if u else None,
        }
        for i, u in rows
    ]


@router.patch("/invitations/{invitation_id}")
def update_invitation(
    invitation_id: str,
    payload: UpdateInviteRequest,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Invitation).filter(Invitation.id == invitation_id).first()
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")
    if inv.is_used:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot update a used invitation")
    if inv.cancelled_at is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot update a cancelled invitation")
    inv.note = payload.note
    inv.invited_email = payload.email.lower().strip() if payload.email else None
    db.commit()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return {
        "id": str(inv.id),
        "code": inv.code,
        "invite_url": f"{FRONTEND_URL}/register?code={inv.code}",
        "note": inv.note,
        "invited_email": inv.invited_email,
        "is_used": inv.is_used,
        "is_cancelled": False,
        "used_at": inv.used_at.isoformat() if inv.used_at else None,
        "cancelled_at": None,
        "expires_at": inv.expires_at.isoformat(),
        "is_expired": inv.expires_at < now,
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
    }


@router.delete("/invitations/{invitation_id}")
def cancel_invitation(
    invitation_id: str,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Invitation).filter(Invitation.id == invitation_id).first()
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")
    if inv.is_used:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot cancel a used invitation")
    if inv.cancelled_at is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invitation is already cancelled")
    inv.cancelled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    return {"message": "Invitation cancelled", "id": invitation_id}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@router.get("/users")
def list_users(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "role": u.role,
            "status": u.status,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        }
        for u in users
    ]


@router.patch("/users/{user_id}/status")
def update_user_status(
    user_id: str,
    payload: UpdateUserStatusRequest,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if payload.status not in ("approved", "suspended"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="status must be 'approved' or 'suspended'")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user.status = payload.status
    db.commit()
    return {"message": f"User status updated to {payload.status}", "user_id": user_id}


# ---------------------------------------------------------------------------
# Broker configs
# ---------------------------------------------------------------------------

@router.get("/broker-configs")
def list_broker_configs(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    configs = (
        db.query(UserBrokerConfig, User)
        .join(User, UserBrokerConfig.user_id == User.id)
        .order_by(UserBrokerConfig.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(cfg.id),
            "user_id": str(cfg.user_id),
            "user_email": user.email,
            "user_name": user.full_name,
            "broker_type": cfg.broker_type,
            "label": cfg.label,
            "account_number": cfg.account_number,
            "is_paper": cfg.is_paper,
            "status": cfg.status,
            "approved_at": cfg.approved_at.isoformat() if cfg.approved_at else None,
            "created_at": cfg.created_at.isoformat() if cfg.created_at else None,
        }
        for cfg, user in configs
    ]


@router.patch("/broker-configs/{config_id}")
def update_broker_config(
    config_id: str,
    payload: UpdateBrokerConfigRequest,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    cfg = db.query(UserBrokerConfig).filter(UserBrokerConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Broker config not found")
    if payload.label is not None:
        cfg.label = payload.label or None
    if payload.account_number is not None:
        cfg.account_number = payload.account_number
    if payload.broker_type is not None:
        cfg.broker_type = payload.broker_type
    if payload.is_paper is not None:
        cfg.is_paper = payload.is_paper
    db.commit()
    return {
        "id": str(cfg.id),
        "user_id": str(cfg.user_id),
        "broker_type": cfg.broker_type,
        "label": cfg.label,
        "account_number": cfg.account_number,
        "is_paper": cfg.is_paper,
        "status": cfg.status,
        "approved_at": cfg.approved_at.isoformat() if cfg.approved_at else None,
        "created_at": cfg.created_at.isoformat() if cfg.created_at else None,
    }


@router.patch("/broker-configs/{config_id}/status")
def update_broker_config_status(
    config_id: str,
    payload: UpdateBrokerConfigStatusRequest,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if payload.status not in ("approved", "rejected"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="status must be 'approved' or 'rejected'")
    cfg = db.query(UserBrokerConfig).filter(UserBrokerConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Broker config not found")
    cfg.status = payload.status
    if payload.status == "approved":
        cfg.approved_by = admin.id
        cfg.approved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    return {"message": f"Broker config {payload.status}", "config_id": config_id}
