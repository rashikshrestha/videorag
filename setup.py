from setuptools import setup, find_packages

setup(
    name="videorag",
    version="0.1.0",
    description="Multimodal RAG system for video temporal grounding",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python-headless>=4.8.0",
        "scenedetect[opencv]>=0.6.3",
        "pysubs2>=1.6.0",
        "sentence-transformers>=2.7.0",
        "transformers>=4.40.0",
        "faiss-cpu>=1.8.0",
        "torch>=2.2.0",
        "pillow>=10.0.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "videorag=scripts.run_pipeline:main",
        ],
    },
)
