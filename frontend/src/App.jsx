import { useState } from 'react'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [prompt, setPrompt] = useState('object')
  const [loading, setLoading] = useState(false)
  const [numDetections, setNumDetections] = useState(0)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultUrl(null)
      setNumDetections(0)
    }
  }

  const handleSegment = async () => {
    if (!selectedImage) {
      alert('Please select an image first')
      return
    }

    setLoading(true)
    setResultUrl(null)

    const formData = new FormData()
    formData.append('file', selectedImage)
    formData.append('prompt', prompt)

    try {
      const response = await fetch('http://localhost:8000/segment-comparison', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Segmentation failed')
      }

      const detections = response.headers.get('X-Detections')
      setNumDetections(parseInt(detections) || 0)

      const blob = await response.blob()
      setResultUrl(URL.createObjectURL(blob))
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to segment image. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
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

      <div>
        <button onClick={handleSegment} disabled={!selectedImage || loading}>
          {loading ? 'Processing...' : 'Segment Image'}
        </button>
      </div>

      {previewUrl && !resultUrl && (
        <div>
          <img src={previewUrl} alt="Preview" style={{ width: '400px' }} />
        </div>
      )}

      {numDetections > 0 && (
        <div>Found {numDetections} object{numDetections !== 1 ? 's' : ''}</div>
      )}

      {resultUrl && (
        <div>
          <img src={resultUrl} alt="Segmentation result" style={{ width: '800px' }} />
        </div>
      )}
    </div>
  )
}

export default App
