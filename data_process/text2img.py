from markdown import markdown  
import subprocess  
from html2image import Html2Image  
import os  
import logging  

# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def markdown_to_html(input_text, save_name, base_dir, temp_dir="temp_mdhtmls", use_default_name=False, css_file='mystyle.css'): 

    cur_dir = os.path.dirname(os.path.abspath(__file__))  
    temp_file_dir = os.path.join(base_dir, temp_dir)
    if not os.path.exists(temp_file_dir):  
        os.makedirs(temp_file_dir)  
    # 如果使用默认 HTML 文件，则将存储在传入路径中  
    save_name = save_name[:-4]
    if use_default_name:  
        md_file_path = os.path.join(temp_file_dir, 'temp.md')  
        html_file_path = os.path.join(temp_file_dir, 'temp.html') 
    else:  
        md_file_path = os.path.join(temp_file_dir, f"{save_name}.md")  
        html_file_path = os.path.join(temp_file_dir, f"{save_name}.html")  
        # 否则，使用默认路径  
 
  
    css_file_path = os.path.join(cur_dir, css_file)  
      
    # 将 Markdown 文本写入临时文件  
    with open(md_file_path, 'w', encoding='utf-8') as md_file:  
        md_file.write(input_text)  
      
    # 构建 pandoc 命令  
    pandoc_command = [  
        'pandoc',  
        '-s',  
        '-c', css_file_path,  
        md_file_path,  
        '-o', html_file_path,  
        '--self-contained'  
    ]  
      
    # 运行 pandoc 命令  
    try:  
        subprocess.run(pandoc_command, check=True)  
        logging.info(f'转换成功: {html_file_path}')  
    except subprocess.CalledProcessError as e:  
        logging.error(f'转换失败: {e}')  
        return None  
      
    return html_file_path  
  
def text_to_image(text, output_image, size=(1280, 1080), save_dir='text_images', data_dir="output", temp_dir="temp_mdhtmls"):  
    """Convert text to image and log the process"""  
    logging.info("Starting the conversion process.")  
  
    # 定义输出路径 
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    output_path = os.path.join(base_dir, save_dir)  
    if not os.path.exists(output_path):  
        os.makedirs(output_path)  

    # Convert text to Markdown (assume the text is in Markdown format)  
    html_file_path = markdown_to_html(text, output_image, base_dir=base_dir, temp_dir=temp_dir, use_default_name=False, )  
    if not html_file_path:  
        return  
 
    # Convert HTML to image using html2image  
    hti = Html2Image(size=size, output_path=output_path, temp_path="/home/lidong1/jianglingjie/temp")  
    # 设置浏览器路径（如果需要）  
    # hti.browser_executable = '/path/to/your/chrome-or-edge'  
  
    hti.screenshot(html_file=html_file_path, save_as=output_image)  
    logging.info(f"Image saved to {os.path.join(output_path, output_image)}.")  
  
    # 清理临时文件  
    # os.remove(md_file_path)  
    # os.remove(html_file_path)  
  
    logging.info("Conversion process completed successfully.")  
  
markdown_text = \
"""Absolutely! Welcome to GPT-ART mode! I'm excited to help you create generative art using Python's turtle graphics. Let's start by setting up a simple turtle environment, and then we'll draw a magnificent piece of art using loops, functions, and some creative experimentation.

Make sure you have the `turtle` module installed for Python. You can install it with:

```python
pip install PythonTurtle
```

Now, let's create a colorful spiral pattern using the turtle module:

```python
import turtle
import random

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")

# Create a turtle
art_turtle = turtle.Turtle()
art_turtle.speed(0)

# Define colors for the art
colors = ["red", "blue", "yellow", "green", "orange", "purple", "white"]

# Function to draw a spiral pattern
def draw_spiral(t, size):
    t.pendown()
    for _ in range(4):
        t.forward(size)
        t.right(90)
    t.penup()
    t.right(10)

# Main loop to create the art
for _ in range(360 // 10):
    art_turtle.color(random.choice(colors))  # Change the color randomly
    size = random.randint(10, 150)  # Change the size randomly
    draw_spiral(art_turtle, size)
    art_turtle.forward(10)
    art_turtle.left(10)

# Hide the turtle and display the art
art_turtle.hideturtle()
turtle.done()
```

This **program** sets up a turtle environment, defines a spiral function, and proceeds to create a colorful spiral pattern by iteratively drawing squares and rotating the turtle artist. The colors and dimensions of the squares are selected randomly, resulting in a unique piece of art every time you run the program.

Feel free to modify the code, experiment with the colors, shapes, and functions to create your own unique artwork!
"""  
  
if __name__ == "__main__":  
    # 示例Markdown文本  
  
    # 调用函数并输出日志  
    output_image = 'output.png'  
    text_to_image(markdown_text, output_image)  
