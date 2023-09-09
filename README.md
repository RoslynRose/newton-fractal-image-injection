## Newton Fractal Image Injection
In this project, we take an arbitrary math function, compute its real and imaginary equivalent function expressions, compute derivative functions' expressions, inject those into an openCL kernel, along with an image. We need inject the image into the first iteration of newton's method then output the result to the frontend.

Install requirements
```bash
pip3 install -r requirements.txt
```

Start the Backend Server
```bash
python3 server.py
```
Then, install the necessary npm packages and start the development server:
```bash
cd reactapp
npm install
npm start
```
