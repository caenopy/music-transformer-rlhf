import os
import glob
import argparse
from jinja2 import Environment, FileSystemLoader

def get_generated_pages(output_dir):
    html_files = glob.glob(f"{output_dir}/*.html")
    return [os.path.basename(file) for file in html_files if os.path.basename(file) != "index.html"]

def get_audio_pairs(directory):
    audio_pairs = []
    supported_extensions = ['mp3']

    for ext in supported_extensions:
        files = glob.glob(f"{directory}/*-A.{ext}")
        for file_a in files:
            file_b = file_a.replace("-A.", "-B.")
            if os.path.exists(file_b):
                audio_pairs.append({
                    'title': os.path.basename(file_a).replace("-A." + ext, ""),
                    'path_a': file_a,
                    'path_b': file_b,
                    'mime_type': f'audio/{ext}'
                })

    return audio_pairs

def main(input_dir, output_dir):
    # Load Jinja2 environment and templates
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    index_template = env.get_template('index_template.html')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the dataset of audio pairs
    audio_pairs = get_audio_pairs(input_dir)

    # Render and save the HTML pages
    for audio_pair in audio_pairs:
        html_content = template.render(
            page_title="Musicality Comparison",
            page_heading="Listen to both audio clips below", #audio_pair['title'],
            audio_pair=audio_pair
        )

        output_file = os.path.join(output_dir, f"{audio_pair['title']}.html")
        with open(output_file, 'w') as f:
            f.write(html_content)

    # Generate index.html with a list of generated pages
    pages = get_generated_pages(output_dir)
    index_html_content = index_template.render(pages=pages)

    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(index_html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HTML pages for audio pairs.')
    parser.add_argument('input_dir', type=str, help='Input directory containing audio pairs.')
    parser.add_argument('output_dir', type=str, help='Output directory for generated HTML pages.')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
