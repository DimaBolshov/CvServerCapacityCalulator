import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import pytz
from enum import Enum
import plotly.graph_objects as go
import pandas as pd

class ImageFormat(Enum):
    """
    Enum for supported image formats with their characteristics.
    The compression_ratio represents typical compression achieved for photo content.
    Base64 encoding increases size by approximately 1.37x due to 64/48 byte ratio.
    """
    JPEG = {'extension': '.jpg', 'compression_ratio': 0.3, 'base64_overhead': 1.37}
    PNG = {'extension': '.png', 'compression_ratio': 0.7, 'base64_overhead': 1.37}
    BASE64 = {'extension': '.b64', 'compression_ratio': 1.0, 'base64_overhead': 1.37}

class ServerLoadCalculator:
    def __init__(self, image_format='JPEG', raw_image_size_kb=500, daily_active_users=4000,
                 frames_per_second=3, photos_per_user=3, max_session_duration=5):
        """
        Initialize calculator with configurable parameters for more flexibility in the Streamlit app
        """
        # User-configurable parameters
        self.DAILY_ACTIVE_USERS = daily_active_users
        self.FRAMES_PER_SECOND = frames_per_second
        self.PHOTOS_PER_USER = photos_per_user
        self.MAX_SESSION_DURATION = max_session_duration * 60  # Convert minutes to seconds
        self.SESSION_INTERVAL = 20
        
        # Time window constants
        self.PEAK_START_HOUR = 17
        self.PEAK_END_HOUR = 2
        self.PEAK_DURATION = (24 - self.PEAK_START_HOUR + self.PEAK_END_HOUR)
        
        # Image parameters
        self.image_format = ImageFormat[image_format.upper()]
        self.raw_image_size_kb = raw_image_size_kb
        self.actual_image_size = self._calculate_actual_image_size()

    def _calculate_actual_image_size(self):
        """
        Calculate actual image size considering format compression and encoding overhead.
        Returns size in kilobytes.
        """
        format_params = self.image_format.value
        compressed_size = self.raw_image_size_kb * format_params['compression_ratio']
        
        # Apply base64 overhead if format is BASE64
        if self.image_format == ImageFormat.BASE64:
            return compressed_size * format_params['base64_overhead']
        return compressed_size

    def calculate_peak_concurrent_users(self):
        """
        Calculate the maximum number of concurrent users during peak hours.
        Uses a probabilistic approach considering user distribution across peak hours.
        """
        # Calculate total session time per user
        total_session_time = (self.MAX_SESSION_DURATION + self.SESSION_INTERVAL) * self.PHOTOS_PER_USER
        
        # Calculate how many users can complete their sessions in an hour
        users_per_hour = 3600 / total_session_time
        
        # During peak hours, we expect more users than average
        peak_hours_users = self.DAILY_ACTIVE_USERS * 0.7  # Assuming 70% of daily users during peak
        
        # Calculate average concurrent users during peak
        avg_concurrent_users = peak_hours_users / (users_per_hour * self.PEAK_DURATION)
        
        # Add 30% buffer for random spikes
        peak_concurrent_users = avg_concurrent_users * 1.3
        
        return int(np.ceil(peak_concurrent_users))
    
    def calculate_photos_per_minute(self):
        """
        Calculate the maximum number of photos processed per minute during peak load.
        Considers the frame rate and number of concurrent users.
        """
        peak_concurrent_users = self.calculate_peak_concurrent_users()
        
        # Calculate frames generated per user per second
        frames_per_user = self.FRAMES_PER_SECOND
        
        # Total frames per second during peak
        total_frames_per_second = peak_concurrent_users * frames_per_user
        
        # Convert to photos per minute
        photos_per_minute = total_frames_per_second * 60
        
        return int(np.ceil(photos_per_minute))

    def calculate_bandwidth_requirements(self):
        """
        Calculate network bandwidth requirements based on image size and transfer rate.
        Returns requirements in Mbps (Megabits per second).
        """
        photos_per_minute = self.calculate_photos_per_minute()
        
        # Convert to bytes per second
        bytes_per_second = (photos_per_minute * self.actual_image_size * 1024) / 60
        
        # Convert to Mbps (Megabits per second)
        mbps = (bytes_per_second * 8) / (1024 * 1024)
        
        # Add 20% overhead for WebSocket headers and other network overhead
        mbps_with_overhead = mbps * 1.2
        
        return mbps_with_overhead

    def calculate_storage_requirements(self):
        """
        Calculate storage requirements for temporary image processing and short-term retention.
        Returns storage needs in GB per hour and per day.
        """
        photos_per_minute = self.calculate_photos_per_minute()
        
        # Calculate hourly storage needs
        hourly_storage_gb = (photos_per_minute * 60 * self.actual_image_size) / (1024 * 1024)
        
        # Calculate daily storage needs (considering peak hours)
        daily_storage_gb = hourly_storage_gb * self.PEAK_DURATION
        
        return {
            'hourly_gb': hourly_storage_gb,
            'daily_gb': daily_storage_gb
        }
    
    def estimate_server_resources(self):
        """
        Estimate server resource requirements based on the calculated load.
        Considers image processing overhead based on format.
        """
        peak_concurrent_users = self.calculate_peak_concurrent_users()
        photos_per_minute = self.calculate_photos_per_minute()
        
        # Adjust resource requirements based on image format
        format_cpu_multiplier = 1.5 if self.image_format == ImageFormat.PNG else 1.0
        
        # Base resource requirements
        MEMORY_PER_PHOTO_MB = 200  # Approximate RAM needed per photo processing
        CPU_CORES_PER_100_PHOTOS = 2 * format_cpu_multiplier
        
        # Calculate required resources
        required_memory_gb = (peak_concurrent_users * MEMORY_PER_PHOTO_MB) / 1024
        required_cpu_cores = np.ceil((photos_per_minute / 100) * CPU_CORES_PER_100_PHOTOS)
        
        return {
            'required_memory_gb': int(np.ceil(required_memory_gb)),
            'required_cpu_cores': int(required_cpu_cores),
            'recommended_servers': int(np.ceil(required_cpu_cores / 32))
        }

def create_hourly_load_chart(calculator):
    """
    Create an interactive chart showing estimated hourly load distribution
    """
    hours = list(range(24))
    peak_users = calculator.calculate_peak_concurrent_users()
    
    # Create hourly load distribution
    load_distribution = []
    for hour in hours:
        if calculator.PEAK_START_HOUR <= hour or hour < calculator.PEAK_END_HOUR:
            load_distribution.append(peak_users)
        else:
            load_distribution.append(peak_users * 0.3)  # Off-peak hours at 30% load
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=load_distribution,
        mode='lines+markers',
        name='Concurrent Users'
    ))
    
    fig.update_layout(
        title='Estimated Hourly Server Load',
        xaxis_title='Hour of Day (MSK)',
        yaxis_title='Concurrent Users',
        height=400
    )
    
    return fig

def main():
    st.set_page_config(page_title="Server Load Calculator", layout="wide")
    
    st.title("Interactive Server Load Calculator")
    st.write("""
    This application helps estimate server requirements for a photo processing system.
    Adjust the parameters below to see how they affect the server load and resource requirements.
    """)
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Parameters")
        daily_users = st.number_input("Daily Active Users", 
                                    min_value=100, 
                                    max_value=100000, 
                                    value=4000)
        
        image_format = st.selectbox("Image Format", 
                                  ["JPEG", "PNG", "BASE64"],
                                  help="Select the image format for processing")
        
        raw_image_size = st.slider("Raw Image Size (KB)", 
                                 min_value=100, 
                                 max_value=2000, 
                                 value=500)
    
    with col2:
        st.subheader("Advanced Parameters")
        frames_per_second = st.slider("Frames per Second", 
                                    min_value=1, 
                                    max_value=10, 
                                    value=3)
        
        photos_per_user = st.slider("Photos per User", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=3)
        
        session_duration = st.slider("Max Session Duration (minutes)", 
                                   min_value=1, 
                                   max_value=15, 
                                   value=5)
    
    # Initialize calculator with user inputs
    calculator = ServerLoadCalculator(
        image_format=image_format,
        raw_image_size_kb=raw_image_size,
        daily_active_users=daily_users,
        frames_per_second=frames_per_second,
        photos_per_user=photos_per_user,
        max_session_duration=session_duration
    )
    
    # Display results in expandable sections
    with st.expander("📊 Load Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Peak Concurrent Users", 
                     calculator.calculate_peak_concurrent_users())
            st.metric("Photos per Minute", 
                     calculator.calculate_photos_per_minute())
        
        with col2:
            bandwidth = calculator.calculate_bandwidth_requirements()
            st.metric("Required Bandwidth (Mbps)", 
                     f"{bandwidth:.2f}")
            st.metric("Recommended Bandwidth (Mbps)", 
                     f"{bandwidth * 1.5:.2f}")
    
    with st.expander("💾 Storage Requirements", expanded=True):
        storage = calculator.calculate_storage_requirements()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Hourly Storage (GB)", 
                     f"{storage['hourly_gb']:.2f}")
        with col2:
            st.metric("Daily Storage (GB)", 
                     f"{storage['daily_gb']:.2f}")
    
    with st.expander("🖥️ Compute Resources", expanded=True):
        resources = calculator.estimate_server_resources()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Required RAM (GB)", 
                     resources['required_memory_gb'])
        with col2:
            st.metric("Required CPU Cores", 
                     resources['required_cpu_cores'])
        with col3:
            st.metric("Recommended Servers", 
                     resources['recommended_servers'])
    
    # Display hourly load distribution chart
    st.subheader("Hourly Load Distribution")
    st.plotly_chart(create_hourly_load_chart(calculator), use_container_width=True)
    
    # Additional recommendations
    st.subheader("📝 Additional Recommendations")
    recommendations = [
        "Implement a queue system for handling traffic spikes",
        "Set up auto-scaling based on concurrent user count",
        "Consider implementing a CDN for global user distribution",
        "Implement efficient cleanup of processed images",
        "Monitor WebSocket connection health",
        "Consider implementing rate limiting per user"
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")

if __name__ == "__main__":
    main()