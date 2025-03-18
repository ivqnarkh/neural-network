import './DrawingCanvas.css'
import {useEffect, useRef, useState} from 'react'

const DrawingCanvas = () => {
    const canvasRef = useRef(null);
    const contextRef = useRef(null)

    const [isDrawing, setIsDrawing] = useState(false)
    const [prediction, setPrediction] = useState(null)

    useEffect(() => {
        const canvas = canvasRef.current;
        canvas.width = 500;
        canvas.height = 500;

        const context = canvas.getContext("2d");
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.lineCap = "round";
        context.strokeStyle = "black";
        context.lineWidth = 15;
        contextRef.current = context;

        document.title = "Digit Classifier";
    }, []);

    const startDrawing = ({nativeEvent}) => {
        const {offsetX, offsetY} = nativeEvent;
        contextRef.current.beginPath();
        contextRef.current.moveTo(offsetX, offsetY);
        contextRef.current.lineTo(offsetX, offsetY);
        contextRef.current.stroke();
        setIsDrawing(true);
        nativeEvent.preventDefault();
    };

    const draw = ({nativeEvent}) => {
        if(!isDrawing) {
            return;
        }
        const {offsetX, offsetY} = nativeEvent;
        contextRef.current.lineTo(offsetX, offsetY);
        contextRef.current.stroke();
        nativeEvent.preventDefault();
    };

    const stopDrawing = () => {
        contextRef.current.closePath();
        setIsDrawing(false);
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
        setPrediction(null);
    }

    const handlePredict = async () => {
        const canvas = canvasRef.current;

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'digit.png');

            try {
                const response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    body: formData,
                });
                
                const data = await response.json();
                setPrediction(data.prediction);
            } catch (error) {
                console.error('Prediction error:', error);
                setPrediction('Error predicting digit');
            }
        }, 'image/png');
    };

    return (
        <div className="canvas-wrapper">
            <h1 className="title">Digit Classifier</h1>
            <div className="content-container">
                <div className="left-panel">
                    <canvas
                        className="canvas-container"
                        ref={canvasRef}
                        onMouseDown={startDrawing}
                        onMouseMove={draw}
                        onMouseUp={stopDrawing}
                        onMouseLeave={stopDrawing}
                    />
                    <div className="controls">
                        <button onClick={handlePredict} className="predict-button">
                            Predict
                        </button>
                        <button onClick={clearCanvas} className="clear-button">
                            Clear
                        </button>
                    </div>
                    {prediction !== null && (
                        <div className="prediction-result">
                            Prediction: {prediction}
                        </div>
                    )}
                </div>
                
                <div className="explanation-box">
                    <h2>Instructions</h2>
                    <p>1. Draw a digit (0-9) in the white canvas area using your mouse.</p>
                    <p>2. Click the "Predict" button to see the AI's prediction.</p>
                    <p>3. Use the "Clear" button to start over.</p>
                    <p>This digit classifier uses a convolutional neural network trained on the MNIST dataset to generate predictions.
                        Make sure your drawing is centered and reasonably sized for best results. This model achieved a testing accuracy of 99%.
                    </p>
                </div>
            </div>
        </div>
    )
}

export default DrawingCanvas