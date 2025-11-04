#!/usr/bin/env python3
"""
简化版脚本：将comparisons目录下最新的比较结果中的PNG文件转换为PDF
只需要Pillow库，不需要PyPDF2
"""

import os
import glob
from PIL import Image
import argparse
from datetime import datetime

def find_latest_comparison_dir(comparisons_dir):
    """找到comparisons目录下最新的比较结果目录"""
    if not os.path.exists(comparisons_dir):
        raise FileNotFoundError(f"Comparisons目录不存在: {comparisons_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(comparisons_dir) 
               if os.path.isdir(os.path.join(comparisons_dir, d)) and not d.startswith('.')]
    
    if not subdirs:
        raise FileNotFoundError(f"在{comparisons_dir}中没有找到比较结果目录")
    
    # 按时间戳排序，获取最新的
    subdirs.sort(reverse=True)
    latest_dir = os.path.join(comparisons_dir, subdirs[0])
    
    print(f"找到最新的比较结果目录: {subdirs[0]}")
    return latest_dir

def find_png_files(directory):
    """在指定目录中查找所有PNG文件"""
    png_pattern = os.path.join(directory, "*.png")
    png_files = glob.glob(png_pattern)
    
    if not png_files:
        print(f"在目录 {directory} 中没有找到PNG文件")
        return []
    
    # 按文件名排序
    png_files.sort()
    
    print(f"找到 {len(png_files)} 个PNG文件:")
    for png_file in png_files:
        print(f"  - {os.path.basename(png_file)}")
    
    return png_files

def convert_png_to_pdf(png_file, output_dir=None):
    """将单个PNG文件转换为PDF"""
    try:
        # 打开PNG图像
        with Image.open(png_file) as img:
            # 如果图像是RGBA模式，转换为RGB
            if img.mode == 'RGBA':
                # 创建白色背景
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 确定输出路径
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                pdf_filename = os.path.join(output_dir, 
                                          os.path.splitext(os.path.basename(png_file))[0] + '.pdf')
            else:
                pdf_filename = os.path.splitext(png_file)[0] + '.pdf'
            
            # 保存为PDF
            img.save(pdf_filename, 'PDF', resolution=100.0)
            print(f"✓ 转换完成: {os.path.basename(png_file)} -> {os.path.basename(pdf_filename)}")
            return pdf_filename
    
    except Exception as e:
        print(f"✗ 转换失败 {os.path.basename(png_file)}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='将comparisons目录下最新比较结果中的PNG文件转换为PDF')
    parser.add_argument('--comparisons-dir', '-d', 
                       default='/Users/ethan/Downloads/Paper/results/comparisons',
                       help='comparisons目录路径 (默认: results/comparisons)')
    parser.add_argument('--output-dir', '-o', 
                       help='PDF输出目录 (默认: 与PNG文件相同目录)')
    
    args = parser.parse_args()
    
    try:
        # 找到最新的比较结果目录
        latest_dir = find_latest_comparison_dir(args.comparisons_dir)
        
        # 查找PNG文件
        png_files = find_png_files(latest_dir)
        
        if not png_files:
            print("没有找到PNG文件，退出")
            return
        
        # 确定输出目录
        output_dir = args.output_dir if args.output_dir else latest_dir
        
        # 转换每个PNG文件
        converted_pdfs = []
        print(f"\n开始转换PNG文件到PDF...")
        print(f"输出目录: {output_dir}")
        print("-" * 50)
        
        for png_file in png_files:
            pdf_file = convert_png_to_pdf(png_file, output_dir)
            if pdf_file:
                converted_pdfs.append(pdf_file)
        
        print("-" * 50)
        print(f"转换完成! 成功转换 {len(converted_pdfs)}/{len(png_files)} 个文件")
        print(f"\n生成的PDF文件:")
        for pdf_file in converted_pdfs:
            print(f"  - {os.path.basename(pdf_file)}")
        
        print(f"\n所有PDF文件位置: {output_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())