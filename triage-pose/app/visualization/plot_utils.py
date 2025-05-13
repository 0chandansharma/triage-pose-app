"""
Utilities for creating plots
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Any, Optional

def create_angle_plot(angles: Dict[str, List[float]], 
                     times: List[float], 
                     title: str = "Joint Angles",
                     figsize: tuple = (8, 6)) -> bytes:
    """
    Create a plot of angle vs time
    
    Args:
        angles: Dictionary of angle names and values
        times: List of time values
        title: Plot title
        figsize: Figure size
        
    Returns:
        bytes: PNG image data
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for angle_name, angle_values in angles.items():
        ax.plot(times, angle_values, label=angle_name)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure to bytes
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    plt.close(fig)
    
    return buf.getvalue()

def create_rom_comparison_chart(current_rom: Dict[str, float], 
                               reference_rom: Dict[str, float], 
                               title: str = "ROM Comparison",
                               figsize: tuple = (10, 6)) -> bytes:
    """
    Create a comparison chart between current ROM and reference values
    
    Args:
        current_rom: Current ROM values for angles
        reference_rom: Reference ROM values for angles
        title: Plot title
        figsize: Figure size
        
    Returns:
        bytes: PNG image data
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar positions
    angles = list(current_rom.keys())
    x = np.arange(len(angles))
    width = 0.35
    
    # Create bars
    current_values = [current_rom[angle] for angle in angles]
    ref_values = [reference_rom.get(angle, 0) for angle in angles]
    
    rects1 = ax.bar(x - width/2, current_values, width, label='Current')
    rects2 = ax.bar(x + width/2, ref_values, width, label='Reference')
    
    # Add labels
    ax.set_xlabel('Joint')
    ax.set_ylabel('Range of Motion (degrees)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(angles, rotation=45, ha='right')
    ax.legend()
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}°',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Save figure to bytes
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    plt.close(fig)
    
    return buf.getvalue()

def create_joint_angle_heatmap(angle_data: pd.DataFrame,
                              title: str = "Joint Angle Heatmap",
                              figsize: tuple = (12, 8)) -> bytes:
    """
    Create a heatmap of joint angles over time
    
    Args:
        angle_data: DataFrame with time and angle data
        title: Plot title
        figsize: Figure size
        
    Returns:
        bytes: PNG image data
    """
    # Exclude time column
    data_for_heatmap = angle_data.drop(columns=['time'])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize data for better visualization
    normalized_data = (data_for_heatmap - data_for_heatmap.min()) / (data_for_heatmap.max() - data_for_heatmap.min())
    
    # Create meshgrid for heatmap
    x = angle_data['time']
    y = np.arange(len(data_for_heatmap.columns))
    X, Y = np.meshgrid(x, y)
    
    # Plot heatmap
    c = ax.pcolormesh(X, Y, normalized_data.T, cmap='viridis', shading='auto')
    
    # Set labels
    ax.set_yticks(np.arange(len(data_for_heatmap.columns)) + 0.5)
    ax.set_yticklabels(data_for_heatmap.columns)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Normalized Angle Value', fontsize=10)
    
    # Save figure to bytes
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    plt.close(fig)
    
    return buf.getvalue()