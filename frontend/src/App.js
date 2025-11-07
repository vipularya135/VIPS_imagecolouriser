import React, { useState } from 'react';

// Use REACT_APP_API_URL in production; fall back to localhost for development
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [file, setFile] = useState(null);
  const [useExample, setUseExample] = useState(false);
  const [loading, setLoading] = useState(false);
  const [images, setImages] = useState({ original: null, single: null, hybrid: null });
  const [error, setError] = useState(null);

  const onFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e && e.preventDefault && e.preventDefault();
    setError(null);
    setImages({ original: null, single: null, hybrid: null });

    const formData = new FormData();
    formData.append('use_example', useExample ? 'true' : 'false');
    if (!useExample && file) {
      formData.append('image', file);
    }

      try {
      setLoading(true);
      const resp = await fetch(`${API_URL}/colorize`, {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) {
        let errMsg = 'Server error';
        try { const err = await resp.json(); errMsg = err.error || errMsg } catch(_){}
        throw new Error(errMsg);
      }
      const data = await resp.json();
      setImages({ original: data.original, single: data.single, hybrid: data.hybrid });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <div className="logo">🎨</div>
          <div>
            <p className="title">Image Colorization</p>
            <p className="subtitle">Normal vs Hybrid Comparison — upload a grayscale image to compare</p>
          </div>
        </div>
        <div className="meta">Backend: <code>{API_URL}</code></div>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit} className="controls">
          <div className="file-input">
            <label className="file-label">
              <input type="file" accept="image/*" onChange={onFileChange} disabled={useExample} />
              <span style={{marginRight:10}}>📁</span>
              <span className="file-name">{file ? file.name : 'Choose an image file...'}</span>
            </label>
          </div>

          <div>
            <label className="checkbox">
              <input type="checkbox" checked={useExample} onChange={(e) => setUseExample(e.target.checked)} />{' '}
              Use example image (vipul.png)
            </label>
          </div>

          <div style={{marginLeft:'auto'}}>
            <button className="cta" type="submit" disabled={loading || (!useExample && !file)}>
              {loading ? <span style={{display:'inline-flex',alignItems:'center',gap:8}}><span className="spinner"/>Processing...</span> : 'Run Colorization'}
            </button>
          </div>
        </form>

        <div className="meta">Compare single-pass and hybrid colorizations. Downloads available after processing.</div>

        {error && <div className="error">{error}</div>}

        {images.original && (
          <section className="results">
            <div className="col">
              <h3>Original</h3>
              <img src={images.original} alt="original" />
              <a className="download" href={images.original} download="original.png">⬇️ Download Original</a>
            </div>
            <div className="col">
              <h3>Colorization</h3>
              <img src={images.single} alt="single" />
              <a className="download" href={images.single} download="colorized_single.png">⬇️ Download Single-Pass</a>
            </div>
            <div className="col">
              <h3>Hybrid Colorization</h3>
              <img src={images.hybrid} alt="hybrid" />
              <a className="download" href={images.hybrid} download="colorized_hybrid.png">⬇️ Download Hybrid</a>
            </div>
          </section>
        )}

        <div className="footer">Tip: Large images may take longer to process. For best results, upload clear grayscale images.</div>
      </div>
    </div>
  );
}

export default App;
