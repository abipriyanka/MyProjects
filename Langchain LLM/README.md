
# News Research Tool 📈

The News Research Tool is a Streamlit web application designed to assist users in analyzing and extracting insights from news articles. This tool leverages natural language processing techniques to process user-provided URLs of news articles, extract relevant information, and provide answers to user queries based on the content of the articles.

## Features

- **URL Input**: Users can input URLs of news articles via the sidebar interface.
- **Article Processing**: The tool processes the provided URLs to load and analyze the content of the news articles.
- **Question Answering**: Users can ask questions about the articles, and the tool retrieves relevant answers from the processed content.
- **Source Identification**: The tool also identifies and displays the sources of the retrieved information.

## Getting Started

To run the News Research Tool locally, you'll need Python and pip installed on your machine.

1. **Clone the Repository**: Clone the repository to your local machine.

    ```bash
    git clone https://github.com/your_username/news-research-tool.git
    ```

2. **Navigate to the Project Directory**: Open a terminal and navigate to the directory where you cloned the repository.

    ```bash
    cd news-research-tool
    ```

3. **Install Dependencies**: Install the required dependencies using pip.

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**: Add your OpenAI API key to the `.env` file in the project directory.

    ```bash
    OPENAI_API_KEY="your_openai_api_key"
    ```

5. **Run the Streamlit App**: Run the Streamlit app using the following command:

    ```bash
    streamlit run app.py
    ```

6. **Input URLs**: Enter the URLs of news articles you want to analyze via the sidebar interface.

7. **Process URLs**: Click the "Process URLs" button to load and process the articles.

8. **Ask Questions**: Enter a question in the text input field provided.

9. **Retrieve Answers**: Click on the "Ask" button to retrieve relevant answers from the processed articles.

## About Streamlit

[Streamlit](https://streamlit.io/) is an open-source Python library that allows you to create beautiful web apps for machine learning and data science projects. With Streamlit, you can build interactive and customizable web applications using simple Python scripts. Streamlit provides easy-to-use components for creating input widgets, visualizations, and interactive elements, making it ideal for rapid prototyping and deployment of data-driven applications.

## Contributing

Contributions to the News Research Tool are welcome! If you encounter any issues, have suggestions for improvements, or would like to contribute new features, feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

