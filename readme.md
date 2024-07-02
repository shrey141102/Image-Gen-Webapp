# AI Image Generation Website

This web application allows users to generate images using various AI models like DALL-E, Lime, and others. Users can input prompts for image generation and view a history of generated images.

## Setup Instructions

### Prerequisites

- Python 3.6+
- pip (Python package installer)
- Git (optional)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <the-repository-url>
   cd <your-local-repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following environment variables:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     LIME_API_KEY=your_lime_api_key
     HORDE_API_KEY=your_horde_api_key
     ```
     Replace `your_openai_api_key`, `your_lime_api_key`, and `your_horde_api_key` with your respective API keys.

### Running the Application

Run the Streamlit application locally:
   ```bash
   streamlit run app.py
   ```

Open your web browser and go to `http://localhost:8501` to view the application.

## Usage Instructions

1. **Select AI Model**: Choose from the dropdown menu the AI model you want to use for image generation.
   
2. **Enter Prompt**: Input a prompt in the text field for generating the image. Example: "A cat playing piano in a forest".

3. **Generate Image**: Click on the "Generate Image" button to initiate the image generation process. The generated image will be displayed on the screen.

4. **View History**: The sidebar displays a history of previously generated images along with their prompts. Scroll down to view the history.

5. **Download Image**: After generating an image, click on the "Download Image" button to download the generated image locally.

6. **Styling**: The application has been styled to center images and ensure a clean display.

---