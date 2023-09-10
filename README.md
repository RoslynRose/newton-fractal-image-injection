## Newton Fractal Image Injection
We process any given mathematical function to extract its real and imaginary components. We then compute the derivative of these expressions. Leveraging OpenCL, we inject these functions into a kernel alongside an image. This image serves as the starting point for our Newton's method iterations. The result is delivered to the frontend for visualization

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
