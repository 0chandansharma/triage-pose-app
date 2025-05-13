# FastAPI application
"""
Main FastAPI application for Triage-Pose
"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os

from app.routers import assessment, exercise, utils, realtime
from app.config import Settings

# Load settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Triage-Pose API",
    description="API for physiotherapy assessment using computer vision",
    version=settings.api_version
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(assessment.router, prefix="/api/v1/assessment", tags=["Assessment"])
app.include_router(exercise.router, prefix="/api/v1/exercise", tags=["Exercise"])
app.include_router(utils.router, prefix="/api/v1/utils", tags=["Utilities"])
app.include_router(realtime.router, prefix="/api/v1/realtime", tags=["Real-time"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Triage-Pose API", "version": settings.api_version}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)