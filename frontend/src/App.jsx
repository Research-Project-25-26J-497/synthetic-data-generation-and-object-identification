import { useState, useEffect, useRef } from 'react';

export default function App() {
  const [isMining, setIsMining] = useState(false);
  const [latestFiles, setLatestFiles] = useState({ latest_json: "None", latest_png: "None", total_datasets: 0 });
  const [systemMessage, setSystemMessage] = useState("System Ready");
  const fileInputRef = useRef(null);

  // Poll the API every 3 seconds to check the Docker container status
  const fetchStatus = async () => {
    try {
      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();
      setIsMining(statusData.is_mining);

      const filesRes = await fetch('/api/latest-files');
      const filesData = await filesRes.json();
      setLatestFiles(filesData);
    } catch (error) {
      console.error("Backend unreachable. Is Uvicorn running?");
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Trigger the Gazebo Docker container
  const startMining = async () => {
    setSystemMessage("Deploying TurtleBot3 into simulation...");
    const res = await fetch('/api/start-mining', { method: 'POST' });
    const data = await res.json();
    setSystemMessage(data.message);
    fetchStatus();
  };

  // Upload a custom .world file via the Hot-Swap API
  const handleWorldUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setSystemMessage(`Uploading ${file.name}...`);
    const res = await fetch('/api/world/upload', { method: 'POST', body: formData });
    const data = await res.json();
    setSystemMessage(data.message);
    fileInputRef.current.value = ""; // Clear the file input
  };

  // Delete the custom world
  const resetWorld = async () => {
    const res = await fetch('/api/world/reset', { method: 'DELETE' });
    const data = await res.json();
    setSystemMessage(data.message);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Header & Status */}
        <div className="flex justify-between items-center bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
          <div>
            <h1 className="text-3xl font-bold text-blue-400">MLOps Lidar Pipeline</h1>
            <p className="text-sm text-gray-400 mt-1">Status: {systemMessage}</p>
          </div>
          <div className={`px-6 py-3 rounded-full font-bold uppercase tracking-wider ${isMining ? 'bg-orange-500 animate-pulse text-white' : 'bg-green-600 text-white'}`}>
            {isMining ? 'Simulation Running...' : 'Idle / Ready'}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Left Column: Control Panel */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 space-y-6">
            <h2 className="text-xl font-semibold border-b border-gray-600 pb-2">Mission Control</h2>
            
            <button 
              onClick={startMining}
              disabled={isMining}
              className={`w-full py-3 rounded font-bold text-white transition-colors ${isMining ? 'bg-gray-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500'}`}
            >
              🚀 Launch Data Miner
            </button>

            <div className="pt-4 space-y-4 border-t border-gray-600">
              <h3 className="text-md font-medium text-gray-300">Environment Hot-Swap</h3>
              <input 
                type="file" 
                accept=".world" 
                onChange={handleWorldUpload} 
                ref={fileInputRef}
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-900 file:text-blue-300 hover:file:bg-blue-800 cursor-pointer"
              />
              <button 
                onClick={resetWorld}
                className="w-full py-2 bg-red-900/50 text-red-400 hover:bg-red-800/50 hover:text-red-300 rounded text-sm transition-colors border border-red-900/50"
              >
                Reset to Default Warehouse
              </button>
            </div>
            
            <div className="pt-4 border-t border-gray-600">
              <p className="text-sm text-gray-400">Total Datasets Generated: <span className="text-white font-mono">{latestFiles.total_datasets}</span></p>
            </div>
          </div>

          {/* Right Column: Visualization */}
          <div className="md:col-span-2 bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Latest Navigation Map</h2>
              {latestFiles.latest_json !== "None" && (
                <a 
                  href="/api/dataset/latest" 
                  download
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold rounded transition-colors"
                >
                  📥 Download Latest JSON
                </a>
              )}
            </div>
            
            <div className="flex-grow bg-gray-900 rounded border border-gray-700 flex items-center justify-center overflow-hidden min-h-[400px]">
              {latestFiles.latest_png !== "None" ? (
                // We add ?t=filename to force the browser to reload the image when a new one is generated!
                <img 
                  src={`/api/map/latest?t=${latestFiles.latest_png}`} 
                  alt="Navigation Map" 
                  className="max-h-[600px] object-contain"
                />
              ) : (
                <p className="text-gray-500 italic">No map generated yet.</p>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2 font-mono">{latestFiles.latest_png}</p>
          </div>

        </div>
      </div>
    </div>
  );
}