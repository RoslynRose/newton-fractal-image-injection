import React, { useState, useEffect, Component } from "react";
import Terminal from './Terminal.js'; // Make sure the path is correct
import axios from "axios";
import "./App.css";
import "./lib/term.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedFile: null,
      processedImage: null,
      string1: '',
      previewURL: null,

    };
  }

  setString1 = (value) => {
    this.setState({ string1: value });
  };

  onFileChange = (event) => {
    const file = event.target.files[0];
    this.setState({
      selectedFile: file,
      previewURL: URL.createObjectURL(file),
    });
  };

  onUpload = async () => {
    this.setState({ uploading: true, processedImage: null });
    const { selectedFile, string1, string2, float1, float2, float3 } = this.state;
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('string1', string1);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        onUploadProgress: (progressEvent) => {
          let percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          this.setState({ progress: percentCompleted });
        },
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      this.setState({ processedImage: response.data.processed_path, uploading: false });
    } catch (error) {
      this.setState({ uploading: false });
      alert('Oops, something went wrong! (⌒_⌒;)');
    }
  };

  render() {
    const { previewURL, processedImage } = this.state;
    return (
      <div className="App">
        <h1>Image Upload with React and Flask</h1>
        <div className="content-wrapper">
          <div>
            <TextInputs 
              setString1={this.setString1} 
            />
            <button onClick={this.onUpload}>Upload</button>
          </div>
          <ImageUploader onFileChange={this.onFileChange} previewURL={previewURL} />
        </div>
        <ProcessedImageView processedImage={processedImage} uploading={this.state.uploading} />
      </div>
    );
  }
}


function ImageUploader({ onFileChange, previewURL }) {
  return (
    <div className="image-uploader">
      <h2>Image Upload:</h2>
      <input type="file" onChange={onFileChange} />
      <div className="image-section">
        {previewURL ? (
          <img src={previewURL} alt="Original" />
        ) : (
          <p>No image selected yet. (・_・;)</p>
        )}
      </div>
    </div>
  );
}

// Updated TextInputs function
function TextInputs({ setString1, setString2, setFloat1, setFloat2, setFloat3 }) {
  return (
    <div className="text-inputs">
      <h2>Function Input</h2>
      <p>Functions should be in the format z * z * z - 1, cos(z*sin(z)) - 1, etc.</p>
      <input type="text" placeholder="Enter string 1" onChange={e => setString1(e.target.value)} />
    </div>
  );
}

// Updated TerminalComponent
class TerminalComponent extends Component {
  componentDidMount() {
    // Assuming 'terminal' is the id of the root element for your terminal
    const terminal = new Terminal('terminal'); 
    terminal.initialize();
    
    terminal.addCommand(
      'activate', 
      (x, y) => `Activating with parameters: ${x}, ${y}`,
      'Usage: activate [x] [y]. Activates something with parameters x and y.'
    );
  }

  render() {
    return (
      <div id="terminal"></div> // Your terminal root element
    );
  }
}


function ProcessedImageView({ processedImage, uploading }) {
  return (
    <div className="content-wrapper">
      <div className="image-section processed">
        {processedImage ? (
          <img src={`http://127.0.0.1:5000${processedImage}`} alt="Processed" />
        ) : uploading ? (
          <span className="rotate-icon">🕑</span>
        ) : (
          <p>No processed image to show. (・_・;)</p>
        )}
      </div>
    </div>
  );
}

export { App, ProcessedImageView };
