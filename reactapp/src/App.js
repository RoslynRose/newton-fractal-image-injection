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
      string2: '',
      float1: 0,
      float2: 0,
      float3: 0,
      previewURL: null,

    };
  }

  // Inside your App component
  setString1 = (value) => {
    this.setState({ string1: value });
  };
  setString2 = (value) => {
    this.setState({ string2: value });
  };
  setFloat1 = (value) => {
    this.setState({ float1: value });
  };
  setFloat2 = (value) => {
    this.setState({ float2: value });
  };
  setFloat3 = (value) => {
    this.setState({ float3: value });
  };


  onFileChange = (event) => {
    const file = event.target.files[0];
    this.setState({
      selectedFile: file,
      previewURL: URL.createObjectURL(file),
    });
  };

  onUpload = async () => {
    const { selectedFile, string1, string2, float1, float2, float3 } = this.state;
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('string1', string1);
    formData.append('string2', string2);
    formData.append('float1', float1);
    formData.append('float2', float2);
    formData.append('float3', float3);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      this.setState({ processedImage: response.data.processed_path });
    } catch (error) {
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
              setString2={this.setString2} 
              setFloat1={this.setFloat1} 
              setFloat2={this.setFloat2} 
              setFloat3={this.setFloat3} 
            />
            <button onClick={this.onUpload}>Upload</button>
          </div>
          <ImageUploader onFileChange={this.onFileChange} previewURL={previewURL} />
        </div>
        <ProcessedImageView processedImage={processedImage} />
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

function ProcessedImageView({ processedImage }) {
  return (
    <div className="content-wrapper">
      <div className="image-section processed">
        {processedImage ? (
          <img src={`http://127.0.0.1:5000${processedImage}`} alt="Processed" />
        ) : (
          <p>No processed image to show. (・_・;)</p>
        )}
      </div>
    </div>
  );
}

export { App, ProcessedImageView };
