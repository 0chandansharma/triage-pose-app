Package the PhysioTrack library:
cd physiotrack
pip install -e . # Install in development mode

# OR

python setup.py bdist_wheel # Build a wheel for distribution

Start the Triage-Pose application:
cd triage-pose
pip install -r requirements.txt
uvicorn app.main:app --reload

Or use Docker to build and run:
cd triage-pose
docker-compose up --build

Then access the application at http://localhost:8000 in your browser.
