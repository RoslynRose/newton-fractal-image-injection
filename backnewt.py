from PIL import Image
import pyopencl as cl
import numpy as np
from sympy import symbols, I, expand, sympify, diff, ccode
import re


def convert_image_to_grayscale(image_path, width, height):
    # Load the image
    img = Image.open(image_path)
    
    # Resize the image
    img = img.resize((width, height))
    
    # Convert to grayscale
    img_gray = img.convert("L")

    # Convert to numpy array and normalize
    img_array = np.array(img_gray).astype(np.uint8)
    img_array = img_array / 255.0

    return img_array


class NewtonFractalGenerator:
    def __init__(self, width=512, height=512, image_path=None):
        self.width = width
        self.height = height

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.fractal_data = np.empty((self.height, self.width), dtype=np.uint16)
        self.fractal_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.fractal_data.nbytes)
        self.final_zx = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zy = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zx_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.final_zx.nbytes)
        self.final_zy_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.final_zy.nbytes)


        # For storing image data and its buffer (if any)
        self.img_array = None
        self.image_buffer = None

        # Load and prepare the image if an image path is given (pretty pics~)
        if image_path:
                self.load_image(image_path)

        # Placeholder for the kernel program
        self.program = None

        # Upload the image data to the GPU (off it goes!)
        cl.enqueue_copy(self.queue, self.image_buffer, self.img_array).wait()

    # ... (all other methods from your original code, but as instance methods)
    # Substitute and Generate Code, Compute Derivatives, Build Program
    def build_program(self, context, kernel_code, expr_str):
        modified_kernel_code = kernel_code.replace('[put it here]', expr_str)
        program = cl.Program(context, modified_kernel_code).build()
        return program

    def substitute_and_generate_code(self, f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y):
        c_code_dict = {
            'f_r': ccode(f_real).replace('x', 'zx').replace('y', 'zy'),
            'f_i': ccode(f_imag).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_r_x': ccode(f_prime_real_x).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_r_y': ccode(f_prime_real_y).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_i_x': ccode(f_prime_imag_x).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_i_y': ccode(f_prime_imag_y).replace('x', 'zx').replace('y', 'zy')
        }

        # Replace re(w) and im(w) since w is real
        for key in c_code_dict:
            c_code_dict[key] = c_code_dict[key].replace('re(w)', 'w').replace('im(w)', '0')
            
        # Fix ambiguous pow calls by explicitly casting the arguments to float
        for key in c_code_dict:
            c_code_dict[key] = re.sub(r'pow\(([^,]+),([^)]+)\)', r'pow((float)\1, (float)\2)', c_code_dict[key])

        c_code_str = f"float f_r = (float)({c_code_dict['f_r']}); "
        c_code_str += f"float f_i = (float)({c_code_dict['f_i']}); "
        c_code_str += f"float f_prime_r_x = (float)({c_code_dict['f_prime_r_x']}); "
        c_code_str += f"float f_prime_r_y = (float)({c_code_dict['f_prime_r_y']}); "
        c_code_str += f"float f_prime_i_x = (float)({c_code_dict['f_prime_i_x']}); "
        c_code_str += f"float f_prime_i_y = (float)({c_code_dict['f_prime_i_y']});"

        return c_code_str

    def build_program(self, kernel_code, expr_str):
        modified_kernel_code = kernel_code.replace('[put it here]', expr_str)
        program = cl.Program(self.context, modified_kernel_code).build()
        return program

    def load_image(self, image_path):
            # Convert the image to grayscale and normalize
            self.img_array = convert_image_to_grayscale(image_path, self.width, self.height)

            # Note: We use float32 here as the image will be used in mathematical operations
            self.img_array = self.img_array.astype(np.float32)

            # Create a buffer for the grayscale image
            self.image_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, self.img_array.nbytes)

            # Upload the image data to the GPU
            cl.enqueue_copy(self.queue, self.image_buffer, self.img_array).wait()



    kernel_code = """
  __kernel void newton_fractal(
        __global ushort *fractal, 
        __global float *image_data,  // Now it's float instead of uchar
        __global float *final_zx,
        __global float *final_zy,
        const unsigned int width, 
        const unsigned int height, 
        const float w)
    {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x;


    //float zx = (float)x / (float)width * 3.5 - 2.0;
    //float zy = (float)y / (float)height * 3.5 - 2.0;
    //float zx = (float)image_data[img_index] / 255.0 * 3.5 - 2.0;
    //float zy = (float)image_data[img_index + 1] / 255.0 * 3.5 - 2.0;

    // Using grayscale intensity for both real and imaginary parts
    float intensity = image_data[index];

    float zx = sin(intensity) * 3.5 - 2.0;
    float zy = cos(intensity) * 3.5 - 2.0;

    ushort iteration = 0;
    const float tolerance = 1.0e-6;

    while (iteration < 100) {
        [put it here]

        // Newton's update: z = z - f(z) / f'(z)
        float denom = f_prime_r_x * f_prime_r_x + f_prime_r_y * f_prime_r_y + f_prime_i_x * f_prime_i_x + f_prime_i_y * f_prime_i_y;
        float zx_new = zx - (f_r * f_prime_r_x + f_i * f_prime_i_x) / denom;
        float zy_new = zy - (f_i * f_prime_r_x - f_r * f_prime_i_x) / denom;

        if ((zx - zx_new) * (zx - zx_new) + (zy - zy_new) * (zy - zy_new) < tolerance * tolerance) break;

        zx = zx_new;
        zy = zy_new;

        iteration++;
    }

    final_zx[index] = zx;
    final_zy[index] = zy;
    fractal[index] = iteration;
}

    """

    def compute_derivatives(self, expr_str, var='z', real_var='x', imag_var='y', param='w'):
        print(f"Debug: expr_str = {expr_str}")  # Add this line for debugging
        x, y, a = symbols(f'{real_var} {imag_var} {param}', real=True)
        z = x + I*y
        expr = sympify(expr_str)
        expr = expr.subs(var, z)
        expr_expanded = expand(expr)
        f_real = expr_expanded.as_real_imag()[0]
        f_imag = expr_expanded.as_real_imag()[1]

        f_prime_real_x = diff(f_real, x)
        f_prime_real_y = diff(f_real, y)
        f_prime_imag_x = diff(f_imag, x)
        f_prime_imag_y = diff(f_imag, y)

        return f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y
    
    def generate_image(self, f, param_value):
        # Computing derivatives and generating GPU code (so technical, nya~)
        f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y = self.compute_derivatives(f)
        c_code_str = self.substitute_and_generate_code(f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y)

        # Build the OpenCL program (big brain time~)
        self.program = self.build_program(self.kernel_code, c_code_str)  # context is already in self, so no need to pass it~


        # Prepare numpy arrays to hold final zx and zy (full of possibilities~)
        final_zx = np.empty((self.height, self.width), dtype=np.float32)
        final_zy = np.empty((self.height, self.width), dtype=np.float32)
        final_zx_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, final_zx.nbytes)
        final_zy_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, final_zy.nbytes)

        # Run the OpenCL program (let the magic happen, nya~)
        if self.image_buffer:  # Check if an image buffer exists
            self.program.newton_fractal(self.queue, self.fractal_data.shape, None, self.fractal_buffer, self.image_buffer, final_zx_buffer, final_zy_buffer, np.uint32(self.width), np.uint32(self.height), np.float32(param_value))

        else:
            self.program.newton_fractal(self.queue, self.fractal_data.shape, None, self.fractal_buffer, final_zx_buffer, final_zy_buffer, np.uint32(self.width), np.uint32(self.height), np.float32(param_value))

        # Finish and fetch data (the treasure at the end of the rainbow~)
        cl.enqueue_copy(self.queue, self.fractal_data, self.fractal_buffer).wait()
        cl.enqueue_copy(self.queue, final_zx, final_zx_buffer).wait()
        cl.enqueue_copy(self.queue, final_zy, final_zy_buffer).wait()

        # Create a colorful image based on final_zx and final_zy
        # For now, I'll just normalize and convert to uint8 for demonstration (simple yet colorful, nya~)
        final_zx = ((final_zx + 2.0) / 3.5 * 255).astype(np.uint8)
        final_zy = ((final_zy + 2.0) / 3.5 * 255).astype(np.uint8)
        print("Debug: final_zx =", final_zx)
        print("Debug: final_zy =", final_zy)


        # Merge channels into a color image (the grand finale~)
        color_image = np.stack([final_zx, np.zeros_like(final_zx), final_zy], axis=2)
        img = Image.fromarray(color_image, 'RGB')

        return img



    def numpy_to_pil(self):
        img = Image.fromarray(self.fractal_data.astype('uint16'))
        img = img.convert("L")
        return img

if __name__ == '__main__':
    # Initialize the fractal generator with the test image "a.jpg"
    fractal_generator = NewtonFractalGenerator(width=1920, height=1920, image_path='a.jpg')

    # Define the function and parameter value
    #f = "sin(cos(z*w)+w)**3 - sin((w/10)**3)*z**4 - sin(w*z*z-w*z+w) - 1"
    f = "z*z*z - 1"
    param_value = 0.01

    # Generate the fractal image
    img = fractal_generator.generate_image(f, param_value)
    
    # Save the generated fractal as "out.jpg"
    img.save("out.jpg")

    print("Fractal image generated and saved as 'out.jpg', nya~ 🌟")
