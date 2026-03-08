# Autonomous LiDAR Data Miner

This repository contains a fully automated, containerized synthetic data generation pipeline for robotics and machine learning research. It runs a headless ROS 2 (Humble) simulation of a TurtleBot3 navigating a custom multi-room warehouse to mine LiDAR and odometry data, all controlled via a modern REST API.

## System Architecture

* **Simulation Environment:** Custom 24x24m multi-room warehouse in Gazebo.
* **Robotics Stack:** ROS 2 Humble operating a TurtleBot3 Waffle Pi.
* **Containerization:** Dockerized headless environment using `xvfb-run` and software rendering to bypass the need for a local GPU.
* **Backend Control:** A FastAPI web server that acts as the control plane, allowing remote triggering and monitoring of the Dockerized simulation.
* **Data Extraction:** Generates timestamped `.json` datasets (X, Y, Yaw, and 360-degree LiDAR arrays) and 2D `.png` floorplan maps generated via Matplotlib.

## Prerequisites

* Ubuntu OS (Tested on 22.04)
* Docker installed and the current user added to the `docker` group.
* Python 3.10+
* FastAPI and Uvicorn (`pip3 install fastapi uvicorn`)

## Installation & Setup

### 1. Clone the repository:
```bash
git clone [https://github.com/Research-Project-25-26J-497/synthetic-data-generation-and-object-identification.git](https://github.com/Research-Project-25-26J-497/synthetic-data-generation-and-object-identification.git)
cd synthetic-data-generation-and-object-identification
```

### 2. Build the Docker Image:
The simulation runs entirely within an isolated Ubuntu/ROS 2 container. You must build this image before starting the API.
```bash
docker build -t lidar-miner:v1 .
```

### 3. Create the Output Directory:
This is where the Docker volume mount will securely teleport the generated datasets out of the container. (If the folder doesnt exist already)

```bash
mkdir -p output
```
## Running the API Server
Start the FastAPI control server using Uvicorn. The server runs directly on the host machine to manage the Docker engine.

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Once running, you can access the interactive API documentation (Swagger UI) at: http://localhost:8000/docs

## API Endpoints
| Method | Endpoint | Description | Response Example |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/start-mining` | Spawns the headless Docker container in the background to collect 1000 LiDAR samples. Includes a global lock to prevent multiple heavy Gazebo instances from crashing the host machine. | `{ "status": "success", "message": "..." }` |
| `GET` | `/api/status` | Used by the frontend to poll the status of the background task. | `{ "is_mining": true/false }` |
| `GET` | `/api/latest-files` | Returns the filenames of the most recently generated JSON dataset and PNG map. | `{ "latest_json": "lidar_dataset_...json", "latest_png": "...", "total_datasets": 5 }` |
| `GET` | `/api/map/latest` | Serves the actual binary `.png` image of the most recently generated exploration map. Can be used directly inside an HTML `<img src="...">` tag. | *Image File (image/png)* |
## Output Format
When a mining run completes, two files are saved to the /output directory:

### 1. JSON Dataset ```(lidar_dataset_[TIMESTAMP].json)```:
* Contains an array of samples. Each sample includes the robot's exact coordinates (odom) and an array of 360 distance measurements (lidar).

### 2. Navigation Map ```(navigation_map_[TIMESTAMP].png)```:
* A high-resolution 2D plot mapping the physical walls of the warehouse (derived from the LiDAR data) and the blue breadcrumb trail of the robot's autonomous exploration path.
