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
"""
\documentclass{article}  
\usepackage{amsmath}  
\begin{document}  
  
根据Plackett-Luce模型你可以使用LaTeX代码在文档中渲染这个公式。以下是完整的LaTeX代码：  
  
```latex  
\documentclass{article}  
\usepackage{amsmath}  
  
\begin{document}  
  
根据Plackett-Luce模型，当K=3时，我们需要考虑三个动作 \(y，当 \(K=3\) 时，\(p^*(y_1 \succ y_2 \succ y_3 | x)\) 的公式如下：  
  
\begin{align}  
    p^*(y_1 \succ y_2 \succ y_3 | x) &= p^*(y_1 | x) \cdot p^*(y_2 | y_1, x)_1, y_2, y_3\) 的排序概率。推导如下：  
  
首先计算 \(y_1\) 在所有动作中被选中的概率：  
  
\[  
p^*(y_1 | x) = \frac{\exp \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x \cdot p^*(y_3 | y_1, y_2, x) \\  
    &= \frac{\exp \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} \right)}{\exp \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} \right) + \exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right) + \exp \left( \beta \)} \right)}{\exp \left( \beta \log \frac{\pi^*(ylog \frac{\pi^*(y_3|x)}{\pi_{\text{ref}}(y_3|x)} \right)} \\  
    &\quad \cdot \frac{\exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)}{\_1|x)}{\pi_{\text{ref}}(y_1|x)} \right) + \exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right) + \exp \left( \beta \log \frac{\pi^*(y_3|x)}{\pi_{\text{ref}}(y_3|x)} \right|x)} \right) + \exp \left( \beta \log \frac{\pi^*(y_3|x)}{\pi_{\text{ref}}(y_3|x)} \right)}  
\]  
  
接下来在剩余的动作中计算 \(y_2\) 被选中的概率（假设 \(y_1\) 已经被选中）：  
  
\[  
p^*(y_2 | y_1, x) = \frac{\exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)}{\exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref)} \\  
    &\quad \cdot 1  
\end{align}  
  
\end{document}  
"""  
  
if __name__ == "__main__":  
    # 示例Markdown文本  
  
    # 调用函数并输出日志  
    output_image = 'output.png'  
    text_to_image(markdown_text, output_image)  
