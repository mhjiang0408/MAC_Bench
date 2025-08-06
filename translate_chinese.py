#!/usr/bin/env python3
"""
Script to remove Chinese comments and translate Chinese text in Python files
"""

import os
import re
import sys
from pathlib import Path

def contains_chinese(text):
    """Check if text contains Chinese characters"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def translate_chinese_comments_and_strings(file_path):
    """Process a Python file to remove/translate Chinese content"""
    
    # Translation mapping for common Chinese phrases
    translations = {
        # Comments
        "# 设置日志": "# Setup logging",
        "# 定义各个子目录路径": "# Define subdirectory paths", 
        "# 检查目录是否存在": "# Check if directory exists",
        "# 创建数据集记录列表": "# Create dataset record lists",
        "# 获取所有期刊名称": "# Get all journal names",
        "# 遍历每个期刊": "# Iterate through each journal",
        "# 获取该期刊下的所有封面": "# Get all covers under this journal",
        "# 遍历每个封面": "# Iterate through each cover",
        "# 提取卷号和期号": "# Extract volume and issue numbers",
        "# 构建各个路径": "# Build various paths",
        "# 检查文件是否存在": "# Check if files exist",
        "# 创建记录": "# Create record",
        "# 添加到记录列表": "# Add to record list",
        "# 如果缺少任何一个文件，添加到空记录列表": "# If missing any file, add to empty records list",
        "# 创建DataFrame并保存": "# Create DataFrame and save",
        "# 确保输出目录存在": "# Ensure output directory exists",
        "# 保存完整数据集": "# Save complete dataset",
        "# 保存空记录数据集": "# Save empty records dataset", 
        "# 打印统计信息": "# Print statistics",
        "# 创建完整记录的DataFrame（排除空记录）": "# Create DataFrame of complete records (excluding empty records)",
        "# 按期刊统计完整记录": "# Statistics of complete records by journal",
        "# 保存期刊统计到CSV": "# Save journal statistics to CSV",
        "# 打印完整记录的期刊统计": "# Print journal statistics for complete records",
        
        # Log messages
        "找到": "Found",
        "个期刊": " journals",
        "处理期刊": "Processing journals",
        "无法解析卷号和期号": "Cannot parse volume and issue numbers",
        "已保存完整数据集到": "Saved complete dataset to",
        "条记录": " records",
        "已保存空记录数据集到": "Saved empty records dataset to", 
        "总记录数": "Total records",
        "完整记录数": "Complete records",
        "不完整记录数": "Incomplete records",
        "各期刊完整记录数": "Complete records by journal",
        "正在处理": "Processing",
        "正在": "Processing",
        "开始": "Starting",
        "完成": "Completed",
        "成功": "Success",
        "失败": "Failed",
        "错误": "Error",
        "警告": "Warning",
        "信息": "Info",
        "结果": "Results",
        "模型": "Model",
        "数据": "Data",
        "文件": "File",
        "路径": "Path",
        "配置": "Config",
        "参数": "Parameter",
        "输出": "Output",
        "输入": "Input",
        "评估": "Evaluation",
        "实验": "Experiment",
        "测试": "Test",
        "训练": "Training",
        "生成": "Generation",
        "分析": "Analysis",
        "解析": "Parse",
        "加载": "Load",
        "保存": "Save",
        "更新": "Update",
        "初始化": "Initialize",
        "计算": "Calculate",
        "处理": "Process",
        "检查": "Check",
        "验证": "Validate",
        "创建": "Create",
        "删除": "Delete",
        "获取": "Get",
        "设置": "Set",
        "执行": "Execute",
        "运行": "Run",
        
        # Function docstrings
        "构建数据集，以Cover为索引，整理出包含所有相关信息的CSV文件": "Construct dataset with Cover as index, organizing all related information into CSV file",
        "包含Article、Cover、Story和Other_Articles文件夹的根目录": "Root directory containing Article, Cover, Story and Other_Articles folders",
        "输出CSV文件的路径": "Output CSV file path",
        "期刊统计CSV文件的路径": "Path to journal statistics CSV file",
        "构建数据集": "Construct the dataset",
        
        # Directory names in error messages
        "Cover目录不存在": "Cover directory does not exist",
        
        # Variable descriptions
        "数据集路径": "dataset path",
        "输出csv文件路径": "output csv file path",
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply translations
        modified_content = content
        for chinese, english in translations.items():
            modified_content = modified_content.replace(chinese, english)
        
        # Remove remaining Chinese characters in comments (lines starting with #)
        lines = modified_content.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and contains_chinese(stripped):
                # Keep the # and spacing, but remove Chinese content
                indent = len(line) - len(line.lstrip())
                processed_lines.append(' ' * indent + '# TODO: Translate this comment')
            else:
                processed_lines.append(line)
        
        final_content = '\n'.join(processed_lines)
        
        # Write back if content changed
        if final_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            print(f"Processed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files"""
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Find all Python files with Chinese content
    python_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Skip this script itself
                if file_path != __file__:
                    python_files.append(file_path)
    
    print(f"Found {len(python_files)} Python files to check")
    
    processed_count = 0
    for file_path in python_files:
        if translate_chinese_comments_and_strings(file_path):
            processed_count += 1
    
    print(f"\nProcessing complete. Modified {processed_count} files.")

if __name__ == "__main__":
    main()