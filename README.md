## Newton Fractal Image Injection
We process any given mathematical function into its real and imaginary equivalent expressions. We then compute the derivative of these expressions. We inject these functions into an OpenCL kernel alongside an image. This image is then injected into the first iteration of the Newton's Method algorithm. The result is delivered to the frontend for visualization.

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
