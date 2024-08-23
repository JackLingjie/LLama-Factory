from selenium import webdriver  
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.chrome.options import Options  
from PIL import Image  
import os  
import markdown  
import logging  
import time
# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def markdown_to_html(markdown_text):  
    """Convert Markdown text to HTML"""  
    html = markdown.markdown(markdown_text)  
    return html  
  
def save_html_to_file(html_content, file_path):  
    """Save HTML content to a file"""  
    with open(file_path, 'w', encoding='utf-8') as file:  
        file.write(html_content)  
  
def html_to_image(html_file, output_image):  
    # Set Chrome options  
    chrome_options = Options()  
    chrome_options.add_argument("--headless")  # Headless mode  
    chrome_options.add_argument("--disable-gpu")  
    chrome_options.add_argument("--no-sandbox")  
      
    # Set ChromeDriver service  
    service = Service('/usr/local/bin/chromedriver')  # Path to ChromeDriver  
      
    # Start Chrome browser  
    driver = webdriver.Chrome(service=service, options=chrome_options)  
      
    # Open HTML file  
    file_url = f"file://{os.path.abspath(html_file)}"  
    driver.get(file_url)  
    time.sleep(2)  
      
    # 调整窗口高度以适应内容  
    total_height = driver.execute_script("return document.body.parentNode.scrollHeight")  
    driver.set_window_size(600, total_height)  
    time.sleep(1)  
      
    # Take screenshot  
    screenshot = driver.get_screenshot_as_png()  
      
    # Save image  
    with open(output_image, 'wb') as file:  
        file.write(screenshot)  
      
    # Close browser  
    driver.quit()  
      
    # Use PIL to adjust image size  
    with Image.open(output_image) as img:  
        img = img.crop(img.getbbox())  # Crop blank areas  
        img.save(output_image)   
  
  
def text_to_image(text, output_image):  
    """Convert text to image and log the process"""  
    logging.info("Starting the conversion process.")  
      
    # Convert text to Markdown (assume the text is in Markdown format)  
    html_content = markdown_to_html(text)  
      
    # Wrap HTML header and style  
    html_content = f"""  
    <!DOCTYPE html>  
    <html lang="en">  
    <head>  
        <meta charset="UTF-8">  
        <title>Markdown to HTML Example</title>  
        <style>  
            body {{  
                font-family: 'Arial', sans-serif;  
            }}  
        </style>  
    </head>  
    <body>  
        {html_content}  
    </body>  
    </html>  
    """  
      
    # Get the directory of the current script  
    script_dir = os.path.dirname(os.path.abspath(__file__))  
      
    # Save HTML to file  
    html_file = os.path.join(script_dir, 'example.html')  
    save_html_to_file(html_content, html_file)  
    logging.info(f"HTML content saved to {html_file}.")  
      
    # Convert HTML to image  
    output_image_path = os.path.join(script_dir, output_image)  
    html_to_image(html_file, output_image_path)  
    logging.info(f"Image saved to {output_image_path}.")  
      
    # Delete HTML file  
    os.remove(html_file)  
    logging.info(f"Temporary HTML file {html_file} deleted.")  
      
    logging.info("Conversion process completed successfully.")  
  
markdown_text = \
"""Developing a daily habit of drawing can be challenging but with consistent practice and a few tips, it can become an enjoyable and rewarding part of your daily routine. Here are some strategies to help you develop the habit of drawing daily:

1. Set a specific time: Allocate a specific time of the day to draw. It could be in the morning, afternoon, or evening. Make drawing a part of your daily routine.
2. Set a specific duration: Determine the amount of time you want to spend on drawing each day. It can be as little as 10 minutes or as long as an hour. Be consistent with the duration to help build the habit.
3. Start small and simple: Don't try to create a masterpiece every day, start with simple and easy-to-do sketches. Focus on improving your skills gradually.
4. Use a variety of tools and mediums: Experiment with different tools like pencils, pens, markers, and different mediums like paper, canvas, or digital apps to keep your drawing practice interesting and engaging.
5. Take breaks and rest: Taking breaks and resting after some time of drawing can help you avoid burnout and stay motivated.
6. Challenge yourself: Set challenges like drawing objects from memory or a specific subject to improve your skills and keep your drawing practice interesting.
7. Track your progress: Keep a record of your daily drawing practice and track your progress. This can be a source of motivation and help you see how far you've come.

Remember, developing a habit takes time and patience. Stay consistent with your drawing practice, be flexible and open to trying new things, and with time, you'll develop a habit of daily drawing that brings you joy and satisfaction. 
""" 
if __name__ == "__main__":
    # 示例Markdown文本  
  
    
    # 调用函数并输出日志  
    output_image = 'output.png'  
    text_to_image(markdown_text, output_image)  
