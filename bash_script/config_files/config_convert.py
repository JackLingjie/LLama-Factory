import yaml  
import re  
  
class MyLoader(yaml.SafeLoader):  
    pass  
  
def construct_yaml_str(self, node):  
    return self.construct_scalar(node)  
  
MyLoader.add_constructor('tag:yaml.org,2002:str', construct_yaml_str)  
  
def yaml_to_args(yaml_file):  
    with open(yaml_file, 'r') as file:  
        config = yaml.load(file, Loader=MyLoader)  
  
    args = []  
  
    def parse_dict(d, prefix=''):  
        for key, value in d.items():  
            if isinstance(value, dict):  
                parse_dict(value, prefix + key + '.')  
            else:  
                if isinstance(value, bool):  
                    if value:  
                        args.append(f"--{prefix}{key}")  
                else:  
                    args.append(f"--{prefix}{key} {value}")  
  
    parse_dict(config)  
  
    return " \\\n".join(args)  
  
yaml_file = 'bash_script/glanchatv2_70B_debug_full_sft_2048_default_template_job.yaml'  
args = yaml_to_args(yaml_file)  
print(args)  
