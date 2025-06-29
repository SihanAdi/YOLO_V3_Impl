"""
读取配置文件
"""

def parse_model_config(file_path):
    """读取模型配置文件"""
    with open(file_path, "r") as f:
        model_defs = []
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                model_defs.append({})
                model_defs[-1]["type"] = line[1:-1].strip()
                if model_defs[-1]["type"] == "convolutional":
                    model_defs[-1]["batch_normalize"] = 0 # 初始化设置默认不使用BN
            else:
                key, value = line.split("=")
                model_defs[-1][key.strip()] = value.strip()
        return model_defs
    
def parse_data_config(file_path):
    options = {}
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(file_path, "r") as f:
        for line in f.readlines():
            items = line.split("=")
            if len(items) != 2:
                continue
            key, value = items[0].strip(), items[1].strip()
            options[key] = value
    return options
            
            
if __name__ == "__main__":
    print(parse_model_config('/Users/adisihansun/Desktop/YOLO_impl/YOLO_V3_Impl/config/yolov3.cfg'))