from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import subprocess
import os
import glob

app = FastAPI(title="Research Bot API")

# GLOBAL LOCK
is_mining_running = False

def check_docker_image_exists(image_name="lidar-miner:v1"):
    """Silently checks if the docker image is built and available locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True
    )
    return result.returncode == 0

def run_docker_miner():
    global is_mining_running
    
    print("--> Background Task: Launching Lidar Miner Container...")
    output_dir = os.path.expanduser("~/ros2_ws/output")
    
    command = [
        "docker", "run", "--rm", 
        "-v", f"{output_dir}:/root/ros2_ws/output", 
        "lidar-miner:v1"
    ]
    
    try:
        # We removed capture_output=True so it prints live to your terminal!
        result = subprocess.run(command)
        if result.returncode == 0:
            print("--> Background Task: Mining complete and container destroyed!")
        else:
            print(f"--> Background Task Error: Container exited with code {result.returncode}")
    except Exception as e:
        print(f"--> Background Task Exception: {e}")
    finally:
        # Always unlock the door
        is_mining_running = False

@app.post("/api/start-mining")
async def start_mining(background_tasks: BackgroundTasks):
    global is_mining_running
    
    # 1. Validation: Check if the Docker image actually exists!
    if not check_docker_image_exists("lidar-miner:v1"):
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "Docker image 'lidar-miner:v1' not found. Please run 'docker build -t lidar-miner:v1 .' in your workspace first."
            }
        )

    # 2. Lock Check: Ensure only one container runs at a time
    if is_mining_running:
        return JSONResponse(
            status_code=409, 
            content={
                "status": "busy",
                "message": "A mining operation is already running. Please wait for it to finish."
            }
        )
        
    # Lock the door BEFORE triggering the background task
    is_mining_running = True 
    background_tasks.add_task(run_docker_miner)
    
    return JSONResponse(content={
        "status": "success",
        "message": "The TurtleBot3 has been deployed in the background."
    })

@app.get("/api/status")
async def get_status():
    """The React Frontend will constantly check this endpoint."""
    return {"is_mining": is_mining_running}

@app.get("/api/latest-files")
async def get_latest_files():
    """Endpoint to check the names of the newest datasets."""
    output_dir = os.path.expanduser("~/ros2_ws/output")
    
    json_files = sorted(glob.glob(f"{output_dir}/*.json"), reverse=True)
    png_files = sorted(glob.glob(f"{output_dir}/*.png"), reverse=True)
    
    return {
        "latest_json": os.path.basename(json_files[0]) if json_files else "None",
        "latest_png": os.path.basename(png_files[0]) if png_files else "None",
        "total_datasets": len(json_files)
    }

# NEW: SERVE THE ACTUAL IMAGE TO THE BROWSER
@app.get("/api/map/latest")
async def get_latest_map():
    """Endpoint to send the actual PNG image file to the frontend."""
    output_dir = os.path.expanduser("~/ros2_ws/output")
    png_files = sorted(glob.glob(f"{output_dir}/*.png"), reverse=True)
    
    # If the output folder is empty, return a standard 404 error
    if not png_files:
        raise HTTPException(status_code=404, detail="No maps have been generated yet.")
        
    latest_png = png_files[0]
    
    # FileResponse automatically sets the correct HTTP headers so the browser knows it's an image!
    return FileResponse(latest_png, media_type="image/png")