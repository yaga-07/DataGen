from setuptools import setup, find_packages

setup(
    name="datagen",
    version="0.1.0",
    description="A minimal yet extensible framework for generating synthetic datasets using LLMs.",
    author="Yash Gajjar",
    author_email="yashgajjar720@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "huggingface_hub",
        "pandas",
        "python-dotenv",
        "google-cloud-aiplatform",
        "google-auth",
        "google-cloud-storage",
        "google-api-python-client",
        "vertexai",
        "google-generativeai",
        "colorama",
        "typing-extensions",
        "duckduckgo-search",
        "beautifulsoup4",
        "requests",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "datagen=main:main"
        ]
    },
)
