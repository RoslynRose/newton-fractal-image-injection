from PIL import Image
import pyopencl as cl
import numpy as np
from sympy import symbols, I, expand, sympify, diff, ccode
from scipy.interpolate import RegularGridInterpolator
import re


class NewtonFractalGenerator:
    def __init__(self, width=512, height=512, image_path=None):
        self.width = width
        self.height = height

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.fractal_data = np.empty((self.height, self.width), dtype=np.uint16)
        self.fractal_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.fractal_data.nbytes)
        self.final_zx = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zy = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zx_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.final_zx.nbytes)
        self.final_zy_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.final_zy.nbytes)

        # For storing image data and its buffer (if any)
        self.img_array = None
        self.image_buffer = None
        self.complex_grid = None
        self.complex_grid_buffer = None

        # Load and prepare the image if an image path is given
        if image_path:
            print("image_path detected")
            self.load_image(image_path)

        # Placeholder for the kernel program
        self.program = None

    kernel_code = """
        __kernel void newton_fractal(
            __global ushort *fractal,
            __global float *complex_grid,  // Replaced image_data with complex_grid
            __global float *final_zx,
            __global float *final_zy,
            const unsigned int width,
            const unsigned int height,
            const float w)
            {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int index = y * width + x;


            float zx = complex_grid[2 * index];
            float zy = complex_grid[2 * index + 1];

            ushort iteration = 0;
            const float tolerance = 1.0e-6;

            while (iteration < 50) {
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
        }"""

    def compute_derivatives(self, expr_str, var='z', real_var='x', imag_var='y', param='w'):
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
        img = Image.open(image_path)
        img = img.resize((self.width, self.height))
        img_gray = img.convert("L")
        image_array = np.array(img_gray).astype(np.float32)  # Changed dtype to float32 for later calculations

        height, width = image_array.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)


        x_new = np.linspace(-1, 1, width)
        y_new = np.linspace(-1, 1, height)
        interpolation_func = RegularGridInterpolator((x, y), image_array)

        xx, yy = np.meshgrid(x_new, y_new)
        grid_points = np.array([xx.flatten(), yy.flatten()]).T
        image_interpolated = interpolation_func(grid_points).reshape(xx.shape)
        image_normalized = image_interpolated / 255.0
        assert image_array.shape == xx.shape, "Shape mismatch between image_array and xx"


        # Create the complex grid
        #self.complex_grid = red_normalized + 1j * blue_normalized * modulation_factor
        modulation_factor = 3.0  # Going for a bold change
        self.complex_grid = np.sin(modulation_factor * xx) + 1j * np.cos(modulation_factor * image_normalized) * np.sin(modulation_factor * yy)


        # Create a buffer for the complex grid
        # We will split the complex grid into real and imaginary parts for the buffer
        self.complex_grid_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, self.complex_grid.nbytes * 2)

        # Prepare a flat array that interleaves the real and imaginary parts
        interleaved_complex_grid = np.empty(self.complex_grid.size * 2, dtype=np.float32)
        interleaved_complex_grid[::2] = self.complex_grid.real.flatten()
        interleaved_complex_grid[1::2] = self.complex_grid.imag.flatten()

        # Upload the complex grid to the GPU
        cl.enqueue_copy(self.queue, self.complex_grid_buffer, interleaved_complex_grid).wait()

    def generate_image(self, f, param_value):
        # Computing derivatives and generating GPU code
        f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y = self.compute_derivatives(f)
        c_code_str = self.substitute_and_generate_code(f_real, f_imag, f_prime_real_x, f_prime_real_y, f_prime_imag_x, f_prime_imag_y)

        # Build the OpenCL program (big brain time~)
        self.program = self.build_program(self.kernel_code, c_code_str)

        # Prepare numpy arrays to hold final zx and zy
        self.final_zx = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zx_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.final_zx.nbytes)
        self.final_zy = np.empty((self.height, self.width), dtype=np.float32)
        self.final_zy_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.final_zy.nbytes)

        # Directly run the OpenCL kernel with self.complex_grid_buffer, this is the magic
        self.program.newton_fractal(self.queue, self.fractal_data.shape, None, 
                                    self.fractal_buffer, self.complex_grid_buffer, 
                                    self.final_zx_buffer, self.final_zy_buffer, 
                                    np.uint32(self.width), np.uint32(self.height), 
                                    np.float32(param_value))


        # Finish and fetch data
        cl.enqueue_copy(self.queue, self.fractal_data, self.fractal_buffer).wait()
        cl.enqueue_copy(self.queue, self.final_zx, self.final_zx_buffer).wait()
        cl.enqueue_copy(self.queue, self.final_zy, self.final_zy_buffer).wait()

        max_iteration = np.max(self.fractal_data)
        normalized_iterations = self.fractal_data / max_iteration

        # Using sin/cos for colors based on the buffers and normalize
        r_normalized = (np.sin(10 * np.pi * normalized_iterations) + 1) / 2
        g_normalized = (np.sin(10 * np.pi * self.final_zx) + 1) / 2
        b_normalized = (np.cos(10 * np.pi * self.final_zy) + 1) / 2

        # Apply gamma correction to enhance brightness
        """
        gamma = 0.5
        r_gamma_corrected = np.power(r_normalized, gamma)
        g_gamma_corrected = np.power(g_normalized, gamma)
        b_gamma_corrected = np.power(b_normalized, gamma)
        """

        def safe_cast_to_uint8(arr):
            arr = np.nan_to_num(arr)  # Replace NaN with 0
            arr = np.where(arr < 0, 0, arr)  # Ensure no value is below 0
            arr = np.where(arr > 255, 255, arr)  # Ensure no value exceeds 255
            return arr.astype(np.uint8)

        r_channel = safe_cast_to_uint8(r_normalized * 255)
        g_channel = safe_cast_to_uint8(g_normalized * 255)
        b_channel = safe_cast_to_uint8(b_normalized * 255)

        #r_channel = (np.clip(r_gamma_corrected * 255, 0, 255)).astype(np.uint8)
        #g_channel = (np.clip(g_gamma_corrected * 255, 0, 255)).astype(np.uint8)
        #b_channel = (np.clip(b_gamma_corrected * 255, 0, 255)).astype(np.uint8)

        # Combine channels to form the color image
        color_image = np.stack([r_channel, g_channel, b_channel], axis=2)

        # Rotating the image 90 degrees to the right
        color_image = np.rot90(color_image, -1)

        img = Image.fromarray(color_image, 'RGB')

        return img

if __name__ == '__main__':
    # Initialize the fractal generator with the test image "a.jpg"
    fractal_generator = NewtonFractalGenerator(width=1920*4, height=1920*4, image_path='a.jpg')

    # Define the function and parameter value
    f = "sin(cos(z*w)+w)**3 - sin((w/10)**3)*z**4 - sin(w*z*z-w*z+w) - 1"
    #f = "sin(cos(z*w)+w)**3 - cos(w*10)*sin((w/10)**3)*z**4 - sin(w*z*z-w*z+w) - 1"
    #f = "z*z*z*z - z*z - 1"
    #f = "z*z*z - 1"
    param_value = 1

    # Generate the fractal image
    img = fractal_generator.generate_image(f, param_value)

    # Save the generated fractal as "out.jpg"
    img.save("out.jpg")

    print("Fractal image generated and saved as 'out.jpg'")

