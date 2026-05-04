from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import glob
import shutil

app = FastAPI(title="Research Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    custom_world_path = os.path.expanduser("~/ros2_ws/custom_world.world")
    
    # Base command with standard output mount
    command = [
        "docker", "run", "--rm", 
        "-v", f"{output_dir}:/root/ros2_ws/output"
    ]
    
    # DYNAMIC HOT-SWAP: If you uploaded a custom world, mount it directly over the default one inside the container!
    if os.path.exists(custom_world_path):
        print("--> Background Task: Custom world detected! Injecting into simulation...")
        
        # Target 1: The compiled runtime folder
        install_path = "/root/ros2_ws/install/research_bot/share/research_bot/worlds/multi_room_warehouse.world"
        
        # Target 2: The raw source folder
        src_path = "/root/ros2_ws/src/research_bot/worlds/multi_room_warehouse.world"
        
        command.extend([
            "-v", f"{custom_world_path}:{install_path}",
            "-v", f"{custom_world_path}:{src_path}"
        ])
        
    command.append("lidar-miner:v1")
    
    try:
        result = subprocess.run(command)
        if result.returncode == 0:
            print("--> Background Task: Mining complete and container destroyed!")
        else:
            print(f"--> Background Task Error: Container exited with code {result.returncode}")
    except Exception as e:
        print(f"--> Background Task Exception: {e}")
    finally:
        is_mining_running = False

@app.post("/api/start-mining")
async def start_mining(background_tasks: BackgroundTasks):
    global is_mining_running
    
    if not check_docker_image_exists("lidar-miner:v1"):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Docker image 'lidar-miner:v1' not found."})

    if is_mining_running:
        return JSONResponse(status_code=409, content={"status": "busy", "message": "A mining operation is already running."})
        
    is_mining_running = True 
    background_tasks.add_task(run_docker_miner)
    
    return JSONResponse(content={"status": "success", "message": "The TurtleBot3 has been deployed in the background."})

@app.get("/api/status")
async def get_status():
    return {"is_mining": is_mining_running}

@app.get("/api/latest-files")
async def get_latest_files():
    output_dir = os.path.expanduser("~/ros2_ws/output")
    json_files = sorted(glob.glob(f"{output_dir}/*.json"), reverse=True)
    png_files = sorted(glob.glob(f"{output_dir}/*.png"), reverse=True)
    
    return {
        "latest_json": os.path.basename(json_files[0]) if json_files else "None",
        "latest_png": os.path.basename(png_files[0]) if png_files else "None",
        "total_datasets": len(json_files)
    }

@app.get("/api/map/latest")
async def get_latest_map():
    output_dir = os.path.expanduser("~/ros2_ws/output")
    png_files = sorted(glob.glob(f"{output_dir}/*.png"), reverse=True)
    if not png_files:
        raise HTTPException(status_code=404, detail="No maps have been generated yet.")
    return FileResponse(png_files[0], media_type="image/png")

# --- NEW ENDPOINTS FOR WORLD MANAGEMENT ---

@app.post("/api/world/upload")
async def upload_world(file: UploadFile = File(...)):
    """Accepts a .world file from the React frontend and saves it for the next simulation run."""
    if not file.filename.endswith(".world"):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid file type. Must be a .world file."})
        
    custom_world_path = os.path.expanduser("~/ros2_ws/custom_world.world")
    
    # Save the uploaded file to the host machine
    with open(custom_world_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"status": "success", "message": f"World '{file.filename}' uploaded successfully! The next mining run will use this geometry."}

@app.delete("/api/world/reset")
async def reset_world():
    """Deletes the custom world so the simulation reverts to your original multi-room warehouse."""
    custom_world_path = os.path.expanduser("~/ros2_ws/custom_world.world")
    if os.path.exists(custom_world_path):
        os.remove(custom_world_path)
        return {"status": "success", "message": "Custom world removed. Reverted to the default multi-room warehouse."}
    return {"status": "success", "message": "Already using the default multi-room warehouse."}

@app.get("/api/dataset/latest")
async def get_latest_dataset():
    """Endpoint to download the actual JSON dataset file."""
    output_dir = os.path.expanduser("~/ros2_ws/output")
    json_files = sorted(glob.glob(f"{output_dir}/*.json"), reverse=True)
    
    # If the output folder is empty, return a 404 error
    if not json_files:
        raise HTTPException(status_code=404, detail="No datasets have been generated yet.")
        
    latest_json = json_files[0]
    
    # FileResponse will trigger a file download in the browser or send raw JSON to a fetch request
    return FileResponse(
        latest_json, 
        media_type="application/json", 
        filename=os.path.basename(latest_json) # This forces the browser to save it with the correct timestamped name
    )