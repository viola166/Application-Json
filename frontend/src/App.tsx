// File: frontend/src/App.tsx
import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Box {
  label: string
  coordinate: [number, number, number, number]
  box_id: number
}

interface LayoutData {
  layout: {
    boxes: Box[]
  }
}

interface FetchedExplanations {
  [frameIndex: number]: {
    [box_id: number]: string; // or whatever type explanation is
  };
}

// component: App
function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [videoWidth, setVideoWidth] = useState<number>(1920)
  const [videoHeight, setVideoHeight] = useState<number>(1080)
  const [displayedVideoSize, setDisplayedVideoSize] = useState<{ width: number; height: number }>({ width: 640, height: 360 })
  const [showGotIt, setShowGotIt] = useState(false);
  const [timeBeforeJump, setTimeBeforeJump] = useState<number | null>(null)
  const [hoveredBoxId, setHoveredBoxId] = useState<number | null>(null)
  const [fetchedExplanations, setFetchedExplanations] = useState<FetchedExplanations>({});
  const [selectedExplanation, setSelectedExplanation] = useState<string | null>(null);
  const [frameIndices, setFrameIndices] = useState<number[]>([])
  const [frameIndex, setFrameIndex] = useState<number>(0)       // frameIndex is current state (initialized as null); setFrameIndex is the function with which you can set a new current state of frameIndex
  const [fps, setFps] = useState<number>(25) 
  const [layoutData, setLayoutData] = useState<LayoutData | null>(null)
  const videoName = "03_05_csp_local_search"
  

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const metadataRes = await fetch(`/metadata/${videoName}`)
        const metadata = await metadataRes.json()
        setFps(metadata.fps || 25)
        setVideoWidth(metadata.width)
        setVideoHeight(metadata.height)

        console.log("fetched metadata")

        const frameRes = await fetch(`/frame/${videoName}/indices`)
        const frameData = await frameRes.json()
        setFrameIndices(frameData)

        console.log("fetched frame indices")
      } catch (err) {
        console.error("Failed to load video metadata or frame indices:", err)
      }
    }

    fetchInitialData()
  }, [videoName])


  useEffect(() => {
    console.log("layoutData updated:", layoutData)
  }, [layoutData])


  // event handler function onPause; defined within App component
  const onPause = async () => {
    const video = videoRef.current
    if (!video) return

    const current_index = Math.floor(video.currentTime * fps)
    let i = 0
    while (i < frameIndices.length && frameIndices[i] < current_index) {
      i += 1
    }

    const next_available_index = frameIndices[i] ?? frameIndices[frameIndices.length - 1]
    setFrameIndex(next_available_index)     

    const rect = video.getBoundingClientRect()
    setDisplayedVideoSize({ width: rect.width, height: rect.height })
      
    const scale = displayedVideoSize.width / videoWidth
    // const scaledVideoHeight = videoHeight * scale
    // const verticalOffset = (displayedVideoSize.height - scaledVideoHeight) / 2

    try {
      const res = await fetch(`/layout/${videoName}/${next_available_index}`)
      const layoutJson = await res.json()
      // for version where I'm returning the json PATH only: handling that frontend and backend are running on different servers
      // const backendBaseUrl = "http://localhost:8000"
      // const layoutJson = await fetch(`${backendBaseUrl}${data.json}`).then(r => r.json())

      const simpleBoxes: Box[] = layoutJson.boxes.map((box: any) => {
        const [x1, y1, x2, y2] = box.coordinate
        return {
          box_id: box.box_id,
          label: box.label,
          coordinate: [
            x1 * scale,
            y1 * scale, //+ verticalOffset,
            x2 * scale,
            y2 * scale //+ verticalOffset,
          ],
        }
      })

      if (layoutJson && layoutJson.boxes) {
        setLayoutData({
          layout: { boxes: simpleBoxes }
        })
      }

    } catch (err) {
      console.error("Failed to fetch layout data:", err)
    }

  }

  const getCurrentTimestamp = () => {
    return videoRef.current ? videoRef.current.currentTime : 0;
  }

  const jumpTo = (timestamp: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = timestamp;  // jump to the specified time
    video.play();                   // continue playing from there
  }

  // after "Explain" Button Click (or indirectly after "Show Context" Button Click)
  const fetchExplanationIfNeeded = async (box_id: number, timestamp: number) => {
    if (fetchedExplanations[frameIndex]?.[box_id]) {
      return fetchedExplanations[frameIndex][box_id];
    }

    const explainResponse = await fetch(`/explain`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json" 
      },
      body: JSON.stringify({ 
        video_name: videoName, 
        timestamp: timestamp, 
        box_id: box_id 
      }),
    });

    if (!explainResponse.ok) throw new Error("Failed to fetch explanation");

    const { explanation } = await explainResponse.json();

    setFetchedExplanations(prev => ({
      ...prev,
      [frameIndex]: {
        ...prev[frameIndex],
        [box_id]: explanation,
      },
    }));

    setSelectedExplanation(explanation);
  };

  // after "Show Context" Button Click
  const handleShowContext = async (box_id: number, timestamp: number) => {
    try {
      const explanation = await fetchExplanationIfNeeded(box_id, timestamp);

      const matchResponse = await fetch("/associate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_name: videoName,
          timestamp: timestamp,
          explanation: explanation,
        }),
      });

      if (!matchResponse.ok) throw new Error("Failed to match explanation");

      const matchData = await matchResponse.json();
      console.log("Matched chunk starts at:", matchData.start);
      console.log("Similarity:", matchData.similarity);
      console.log("Label:", matchData.label);
      // Do something useful with context...

      if (videoRef.current) {
        jumpTo(matchData.start)
      }
      
      setShowGotIt(true)
      setTimeBeforeJump(timestamp)

    } catch (error) {
      console.error("Error while fetching context:", error);
    }
  };

  const onGotItClick = () => {
    const video = videoRef.current;
    if (!video) return;

    if (timeBeforeJump !== null) {
      video.currentTime = timeBeforeJump - 5;
      video.play();
    }
    setShowGotIt(false);
  };

  return (
    <div className='main-container'>
      <div className='video-container'>
        <video className='video-player'
          ref={videoRef}
          width="640"
          controls
          onPause={onPause}
          onPlay={() => {
            setLayoutData(null)
            setHoveredBoxId(null)
          }}
          style={{ display: 'block' }}
        >
          <source src={`/video/${videoName}`} type="video/mp4" />
        </video>

        {layoutData && (
          <div className="layout-overlay">
            {layoutData.layout.boxes.map((box, idx) => (
              <div className="overlay-box"
                key={idx}
                style={{
                  top: box.coordinate[1],
                  left: box.coordinate[0],
                  width: box.coordinate[2] - box.coordinate[0],
                  height: box.coordinate[3] - box.coordinate[1],
                }}
                onMouseEnter={() => setHoveredBoxId(box.box_id)}
                onMouseLeave={() => setHoveredBoxId(null)}
              >
                {hoveredBoxId === box.box_id && (
                  <div className='overlay-highlight'>
                    <div className='button-background'
                        style = {{
                          width: displayedVideoSize.width * 0.15,
                          height: displayedVideoSize.height * 0.15,
                        }}
                    >
                      <button className='Get-AI-Mentor'
                        onClick={(e) => {
                          e.stopPropagation();
                          fetchExplanationIfNeeded(box.box_id, getCurrentTimestamp());
                        }}
                      >
                        Explain
                      </button>
                      <button className='Get-AI-Mentor'
                        onClick={(e) => {
                          e.stopPropagation();
                          handleShowContext(box.box_id, getCurrentTimestamp());
                        }}
                      >
                        Show Context
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {showGotIt && (
          <button className='navigator-button'
            onClick={onGotItClick}
          >
            Got it!
          </button>
        )}
      </div>
      {selectedExplanation && (
        <div className="chat-panel"
          style= {{
            height: displayedVideoSize.height - 30,     /* -30 because of padding 15px*/
          }}
        >
          <div className="chat-header">
            <h3>AI Mentor</h3>
            <button onClick={() => setSelectedExplanation(null)}>✕</button>
          </div>
          <p>{selectedExplanation}</p>
        </div>
      )}
    </div>

  )
}

export default App
