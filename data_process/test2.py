from markdown import markdown  
from html2image import Html2Image  
import os  
import logging  
  
# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def markdown_to_html(markdown_text):  
    """Convert Markdown text to HTML"""  
    html = markdown(markdown_text)  
    return html  
  
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
                margin: 0;  
                padding: 20px;  
                background-color: white;  
                color: black;  
            }}  
        </style>  
    </head>  
    <body>  
        {html_content}  
    </body>  
    </html>  
    """  
    output_path = os.path.dirname(os.path.abspath(__file__)) 
    # Convert HTML to image using html2image  
    hti = Html2Image(size=(800, 600), temp_path="/home/lidong1/jianglingjie/temp", output_path=output_path)  

    # 设置浏览器路径（如果需要）  
    # hti.browser_executable = '/path/to/your/chrome-or-edge'  
     # Get the directory of the current script  
    # script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # output_image = os.path.join(script_dir, output_image)    
    # output_image = 'data_process/red_page.png'
    # logging.info(f"Image saved to {output_image}.")    
    hti.screenshot(html_str=html_content, save_as=output_image)  
      
    logging.info(f"Image saved to {os.path.join(output_path, output_image)}.")  
      
    logging.info("Conversion process completed successfully.")  
  
markdown_text = """  
# Hello, World!  
This is an example document written using **Markdown**.  
- Item 1  
- Item 2  
- Item 3  
[Click here](http://example.com) to visit the example website.  
"""  
  
if __name__ == "__main__":  
    # 示例Markdown文本  
      
    # 调用函数并输出日志  
    output_image = 'output.png'  
    text_to_image(markdown_text, output_image)  
