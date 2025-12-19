import { useState } from 'react'

function App() {
  const [mode, setMode] = useState('image')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [prompt, setPrompt] = useState('object')
  const [loading, setLoading] = useState(false)
  const [numDetections, setNumDetections] = useState(0)
  const [videoInfo, setVideoInfo] = useState(null)
  const [startFrame, setStartFrame] = useState(0)
  const [endFrame, setEndFrame] = useState(0)
  const [loadingInfo, setLoadingInfo] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultUrl(null)
      setNumDetections(0)

      if (file.type.startsWith('video/')) {
        setLoadingInfo(true)
        try {
          const formData = new FormData()
          formData.append('file', file)
          const response = await fetch('http://localhost:8000/get-video-info', {
            method: 'POST',
            body: formData,
          })
          const info = await response.json()
          console.log('Video info:', info)
          setVideoInfo(info)
          setStartFrame(0)
          setEndFrame(info.total_frames)
        } catch (error) {
          console.error('Error getting video info:', error)
          alert('Failed to load video info')
        } finally {
          setLoadingInfo(false)
        }
      } else {
        setVideoInfo(null)
      }
    }
  }

  const handleSegment = async () => {
    if (!selectedFile) {
      alert('Please select a file first')
      return
    }

    setLoading(true)
    setResultUrl(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('prompt', prompt)

    try {
      let response
      if (mode === 'image') {
        console.log('Calling segment-comparison')
        response = await fetch('http://localhost:8000/segment-comparison', {
          method: 'POST',
          body: formData,
        })
      } else {
        const sessionId = Date.now().toString()
        formData.append('start_frame', startFrame.toString())
        formData.append('end_frame', endFrame.toString())
        formData.append('session_id', sessionId)
        console.log('Calling segment-video with:', { prompt, startFrame, endFrame, sessionId })
        
        setProgress({ current: 0, total: endFrame - startFrame, status: 'processing' })
        
        const progressInterval = setInterval(async () => {
          try {
            const progressResponse = await fetch(`http://localhost:8000/progress/${sessionId}`)
            const progressData = await progressResponse.json()
            console.log('Progress update:', progressData)
            setProgress(progressData)
            if (progressData.status === 'complete' || progressData.status === 'not_found') {
              clearInterval(progressInterval)
            }
          } catch (error) {
            console.error('Progress fetch error:', error)
          }
        }, 300)
        
        response = await fetch('http://localhost:8000/segment-video', {
          method: 'POST',
          body: formData,
        })
        clearInterval(progressInterval)
        console.log('Response status:', response.status)
      }

      if (!response.ok) {
        throw new Error('Segmentation failed')
      }

      const detections = response.headers.get('X-Detections')
      if (detections) {
        setNumDetections(parseInt(detections) || 0)
      }

      const blob = await response.blob()
      setResultUrl(URL.createObjectURL(blob))
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to segment. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div>
        <button onClick={() => setMode('image')}>Image</button>
        <button onClick={() => setMode('video')}>Video</button>
      </div>

      <div>
        <input
          type="file"
          accept={mode === 'image' ? 'image/*' : 'video/*'}
          onChange={handleFileChange}
        />
      </div>

      <div>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., dog, car, person"
        />
      </div>

      {mode === 'video' && loadingInfo && (
        <div>Loading video info...</div>
      )}

      {mode === 'video' && videoInfo && (
        <div>
          <div>Total frames: {videoInfo.total_frames} | FPS: {videoInfo.fps} | Resolution: {videoInfo.width}x{videoInfo.height}</div>
          <div>
            Start frame: 
            <input
              type="range"
              min="0"
              max={videoInfo.total_frames}
              value={startFrame}
              onChange={(e) => setStartFrame(parseInt(e.target.value))}
            />
            <input
              type="number"
              min="0"
              max={videoInfo.total_frames}
              value={startFrame}
              onChange={(e) => {
                const val = e.target.value === '' ? 0 : parseInt(e.target.value)
                setStartFrame(val)
              }}
              style={{ width: '80px', marginLeft: '10px' }}
            />
          </div>
          <div>
            End frame: 
            <input
              type="range"
              min="0"
              max={videoInfo.total_frames}
              value={endFrame}
              onChange={(e) => setEndFrame(parseInt(e.target.value))}
            />
            <input
              type="number"
              min="0"
              max={videoInfo.total_frames}
              value={endFrame}
              onChange={(e) => {
                const val = e.target.value === '' ? 0 : parseInt(e.target.value)
                setEndFrame(val)
              }}
              style={{ width: '80px', marginLeft: '10px' }}
            />
          </div>
          <div>Processing {endFrame - startFrame} frames</div>
        </div>
      )}

      <div>
        <button onClick={handleSegment} disabled={!selectedFile || loading}>
          {loading ? 'Processing...' : `Segment ${mode}`}
        </button>
        {loading && mode === 'video' && (
          <div>
            {progress.total > 0 ? (
              <>
                Processing frame {progress.current} / {progress.total} ({Math.round((progress.current / progress.total) * 100)}%)
                <br />
                Estimated time: {Math.round((progress.total - progress.current) * 2)} seconds remaining
              </>
            ) : (
              'Starting...'
            )}
          </div>
        )}
      </div>

      {previewUrl && !resultUrl && (
        <div>
          {mode === 'image' ? (
            <img src={previewUrl} alt="Preview" style={{ width: '400px' }} />
          ) : (
            <video src={previewUrl} controls style={{ width: '400px' }} />
          )}
        </div>
      )}

      {numDetections > 0 && (
        <div>Found {numDetections} object{numDetections !== 1 ? 's' : ''}</div>
      )}

      {resultUrl && (
        <div>
          {mode === 'image' ? (
            <img src={resultUrl} alt="Result" style={{ width: '800px' }} />
          ) : (
            <video src={resultUrl} controls style={{ width: '800px' }} />
          )}
        </div>
      )}
    </div>
  )
}

export default App
