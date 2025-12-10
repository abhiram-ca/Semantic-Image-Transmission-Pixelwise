import React, { useState, useRef } from 'react';
import { Upload, Wifi, ImageIcon, Activity, Info, CheckCircle, XCircle, Loader } from 'lucide-react';
import './App.css';

export default function SemanticCommApp() {
  const [image, setImage] = useState(null);
  const [snr, setSnr] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const fileInputRef = useRef(null);

  React.useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health');
      const data = await response.json();
      setBackendStatus(data.model_loaded ? 'connected' : 'error');
    } catch (err) {
      setBackendStatus('disconnected');
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!image) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: image,
          snr: snr
        })
      });

      const data = await response.json();

      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || 'Processing failed');
      }
    } catch (err) {
      setError(`Connection error: ${err.message}. Make sure backend is running on port 5000.`);
    } finally {
      setLoading(false);
    }
  };

  const StatusBadge = () => {
    const statusConfig = {
      checking: { 
        className: 'status-checking',
        icon: <Loader className="spin" size={16} />,
        text: 'Checking...' 
      },
      connected: { 
        className: 'status-connected',
        icon: <CheckCircle size={16} />,
        text: 'Connected' 
      },
      disconnected: { 
        className: 'status-disconnected',
        icon: <XCircle size={16} />,
        text: 'Disconnected' 
      },
      error: { 
        className: 'status-error',
        icon: <XCircle size={16} />,
        text: 'Model Error' 
      }
    };

    const config = statusConfig[backendStatus];

    return (
      <div className={`status-badge ${config.className}`}>
        {config.icon}
        <span>{config.text}</span>
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        
        {/* Header */}
        <div className="header-card">
          <div className="header-content">
            <div className="header-left">
              <div className="header-icon">
                <Wifi size={32} />
              </div>
              <div>
                <h1 className="main-title">
                  Semantic Communication Network
                </h1>
                <p className="subtitle">
                  Deep learning-based image transmission with semantic compression
                </p>
              </div>
            </div>
            <StatusBadge />
          </div>
        </div>

        {/* Info Panel */}
        <div className="info-panel">
          <div className="info-content">
            <div className="info-icon">
              <Info size={20} />
            </div>
            <div className="info-text">
              <h3>How it works</h3>
              <p>
                This system extracts semantic features from images, transmits them through a noisy channel, 
                and reconstructs the image at the receiver. The SNR parameter controls channel quality.
              </p>
            </div>
          </div>
        </div>

        <div className="main-grid">
          
          {/* Left Panel - Controls */}
          <div className="controls-panel">
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-header-icon">
                  <Activity size={20} />
                </div>
                <h2>Controls</h2>
              </div>

              {/* Image Upload */}
              <div className="control-group">
                <label className="control-label">Upload Image</label>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept="image/*"
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="upload-button"
                >
                  <Upload size={22} />
                  <span>Choose Image</span>
                </button>
                {image && (
                  <div className="preview-container">
                    <img src={image} alt="Uploaded" className="preview-image" />
                  </div>
                )}
              </div>

              {/* SNR Control */}
              <div className="control-group">
                <label className="control-label">Signal-to-Noise Ratio (SNR)</label>
                <div className="snr-control">
                  <div className="snr-slider-container">
                    <input
                      type="range"
                      min="-5"
                      max="100"
                      step="0.5"
                      value={snr}
                      onChange={(e) => setSnr(parseFloat(e.target.value))}
                      className="snr-slider"
                    />
                    <span className="snr-value">{snr} dB</span>
                  </div>
                  <div className="snr-labels">
                    <span>Poor (-5)</span>
                    <span>Excellent (30)</span>
                  </div>
                </div>
              </div>

              {/* Process Button */}
              <button
                onClick={processImage}
                disabled={!image || loading || backendStatus !== 'connected'}
                className="process-button"
              >
                {loading ? (
                  <>
                    <Loader className="spin" size={22} />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Wifi size={22} />
                    <span>Transmit & Reconstruct</span>
                  </>
                )}
              </button>

              {/* Error Display */}
              {error && (
                <div className="error-box">
                  <XCircle size={18} />
                  <p>{error}</p>
                </div>
              )}

              {/* Metrics */}
              {results && (
                <div className="metrics-container">
                  <h3>Quality Metrics</h3>
                  <div className="metrics-list">
                    <div className="metric-card metric-psnr">
                      <span>PSNR</span>
                      <strong>{results.metrics.psnr.toFixed(2)} dB</strong>
                    </div>
                    {results.metrics.ssim !== null && (
                      <div className="metric-card metric-ssim">
                        <span>SSIM</span>
                        <strong>{results.metrics.ssim.toFixed(4)}</strong>
                      </div>
                    )}
                    <div className="metric-card metric-cosine">
                      <span>Cosine Similarity</span>
                      <strong>{results.metrics.cosine_similarity.toFixed(4)}</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="results-panel">
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-header-icon">
                  <ImageIcon size={20} />
                </div>
                <h2>Results</h2>
              </div>

              {!results && (
                <div className="empty-results">
                  <div className="empty-icon">
                    <Wifi size={64} />
                  </div>
                  <p className="empty-text">Upload an image and click "Transmit & Reconstruct"</p>
                  <p className="empty-subtext">Your results will appear here</p>
                </div>
              )}

              {results && (
                <div className="results-content">
                  

                  {/* Individual Images */}
                  <div className="images-grid">
                    <div className="image-card">
                      <h3>Original</h3>
                      <img src={results.original} alt="Original" className="result-image" />
                    </div>
                    <div className="image-card">
                      <h3>Semantic Map</h3>
                      <img src={results.semantic} alt="Semantic" className="result-image-small" />
                    </div>
                    <div className="image-card">
                      <h3>Reconstructed</h3>
                      <img src={results.reconstructed} alt="Reconstructed" className="result-image-small" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="instructions-card">
          <h2>Setup Instructions</h2>
          <div className="instructions-grid">
            <div className="instruction-card instruction-1">
              <div className="instruction-number">1</div>
              <h3>Backend Setup</h3>
              <ul>
                <li>Install Python 3.7+</li>
                <li>Create a virtual environment</li>
                <li>Run: <code>pip install -r requirements.txt</code></li>
                <li>Start: <code>python app.py</code></li>
              </ul>
            </div>

            <div className="instruction-card instruction-2">
              <div className="instruction-number">2</div>
              <h3>Usage</h3>
              <ul>
                <li>Upload an image</li>
                <li>Adjust SNR slider</li>
                <li>Click "Transmit & Reconstruct"</li>
                <li>View results and metrics</li>
              </ul>
            </div>

            <div className="instruction-card instruction-3">
              <div className="instruction-number">3</div>
              <h3>Metrics</h3>
              <ul>
                <li><strong>PSNR:</strong> Higher is better</li>
                <li><strong>SSIM:</strong> 0-1 scale (closer to 1 better)</li>
                <li><strong>Cosine Sim:</strong> Feature similarity (0-1)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}