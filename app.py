import gradio as gr
import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
from utils.util import move_to_device
from utils.camera import generate_circle_poses, generate_lgm_poses, get_proj_matrix, convert_to_opencv
from data.mv_input_data import prepare_real_data, prepare_gso_data
from data.diffusion_data import prepare_crm_data, prepare_imagegen_data, setup_crm_diffusion, setup_imagegen_diffusion
from models.compute_graph.graph_ccm_stage_2_gs_square import Graph
import imageio
from torchvision.utils import save_image
from einops import rearrange
import utils.options as options
import utils.util_vis as util_vis
from utils.options import gradio_set
import plotly.graph_objects as go
import trimesh

YAML = 'options/demo.yaml'

# Set options and load model once
print('[DEBUG] Loading options from YAML file...')
opt = gradio_set(opt_fname=YAML)
opt.device = 0  # or "cuda" if applicable
print(f'[DEBUG] Setting device to: {opt.device}')
graph = Graph(opt).to(opt.device)
graph.eval()
print('[DEBUG] Model loaded and set to evaluation mode.')

# Assuming your `gen` function remains unchanged, as defined above
def gen(opt, graph, pipeline=None):
    print('[DEBUG] Starting generation process...')
    if opt.image_data:
        print('[INFO] Using random image input data ...')
        data_list, name_list, load_path = prepare_real_data(opt)
    elif opt.single_input:
        if opt.crm:
            print('[INFO] Using mv diffusion from CRM...')
            data_list, name_list, load_path = prepare_crm_data(opt, pipeline)
        else:
            print('[INFO] Using MV dream...')
            data_list, name_list, load_path = prepare_mvdream_data(opt, pipeline)
    else:
        # gso demo in paper, no need to rmbg
        print('[INFO] Using demo GSO data ...')
        data_list, name_list, load_path = prepare_gso_data(opt)
    print(f'[DEBUG] Data loaded from: {load_path}')

    if opt.lucid_cam:
        print(f'[INFO] Using camera orbit from Lucid Dreamer.... ')
        poses = generate_circle_poses(size=150) # 150, 4, 4
    else:
        print('[INFO] Using camera orbit from LGM... ')
        poses = generate_lgm_poses() # 180, 4, 4
    print('[DEBUG] Camera poses generated.')

    proj_matrix = get_proj_matrix()
    gs_cam_view, gs_cam_view_proj, gs_cam_pos = convert_to_opencv(poses, proj_matrix)
    data_list[0].gs_cam_view = gs_cam_view.unsqueeze(0)
    data_list[0].gs_cam_view_proj = gs_cam_view_proj.unsqueeze(0)
    data_list[0].gs_cam_pos = gs_cam_pos.unsqueeze(0)
    print('[DEBUG] Camera views and projection matrices converted.')

    # create the save dir
    save_folder = os.path.join(load_path, 'preds')
    if os.path.isdir(save_folder):
        print(f'[DEBUG] Removing existing save folder: {save_folder}')
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    print(f'[DEBUG] Save folder created: {save_folder}')

    if opt.save_frames:
        frame_folder = os.path.join(save_folder, 'frames')
        if os.path.isdir(frame_folder):
            print(f'[DEBUG] Removing existing frame folder: {frame_folder}')
            shutil.rmtree(frame_folder)
        os.makedirs(frame_folder)
        print(f'[DEBUG] Frame folder created: {frame_folder}')

    opt.output_path = load_path
    print(f'[DEBUG] Output path set to: {opt.output_path}')

    # inference the model and save the results
    progress_bar = tqdm(data_list)
    for i, var in enumerate(progress_bar):
        print(f'[DEBUG] Processing data item {i}...')
        # forward
        with torch.no_grad():
            var = move_to_device(var, opt.device)
            print(f'[DEBUG] Data moved to device: {opt.device}')
            var = graph.forward(opt, var, training=False, get_loss=False, ccm_only=False)
            print('[DEBUG] Forward pass completed.')
            if 'seen_points_pred' in var:
                _filename_pred = ('ccm_input', 'seen_surface_pred')
                util_vis.warp_vis_function(opt, var.idx, _filename_pred, var.seen_points_pred, var.rgb_input_map[0], folder='preds')
                print('[DEBUG] Seen points prediction visualized.')
            if opt.save_per_view_ply:
                _fname = 'per_view_ply'
                for i, ply in enumerate(var.seen_points_pred):
                    util_vis.vis_per_view_ply(opt, i, _fname, ply.unsqueeze(0), var.rgb_input_map[0][i:i+1], folder='preds')
                    print(f'[DEBUG] Per-view PLY saved for index {i}.')
            if 'pred_images' in var and opt.save_frames:
                pred_frames = var.pred_images.reshape(-1, 3, 512, 512)
                for i, img in enumerate(pred_frames):
                    util_vis.dump_gs_images(opt, i, 'gs_rendering', img.unsqueeze(0), None, folder='preds/frames')
                    print(f'[DEBUG] Predicted frame saved for index {i}.')
            if 'opacity' in var:
                conf_map = rearrange(var.opacity, 'b (v h w) c -> (b v) c h w', h=256, v=var.rgb_input_map.shape[1])
                fn = opt.output_path.split('/')[-1]
                _file_path = "{}/{}/{}_{}.png".format(opt.output_path, 'preds', fn, 'conf_map')
                save_image(conf_map, _file_path)
                print('[DEBUG] Opacity map saved.')
            if opt.save_video:
                fn = opt.output_path.split('/')[-1]
                frames = []
                for p in var.pred_images.reshape(-1, 3, 512, 512):
                    p = torch.clamp(p, 0.0, 1.0)
                    p = p.permute(1,2,0).detach().cpu().numpy()
                    p = (p * 255).round().astype('uint8')
                    frames.append(p)
                _file_path = "{}/{}/{}_{}.mp4".format(opt.output_path, 'preds', fn, 'video_rgb')
                imageio.mimwrite(_file_path, frames, fps=30, quality=8)
                print('[DEBUG] Video saved.')
                
    print('==> results saved at folder: {}/preds'.format(opt.output_path))

# Predefined examples directory
EXAMPLES_DIR = "examples"

# List predefined examples from the "examples" folder
def list_example_options():
    return [folder for folder in os.listdir(EXAMPLES_DIR) if os.path.isdir(os.path.join(EXAMPLES_DIR, folder))]

# Function to display example images
def display_example_images(example_name):
    input_path = os.path.join(EXAMPLES_DIR, example_name)
    images = []

    # Traverse the folder and look for images, skipping "mask" subfolders
    for root, dirs, files in os.walk(input_path):
        # Skip folders named "mask"
        dirs[:] = [d for d in dirs if d.lower() != "masks"]
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, img_file))

    return images

# Function to provide dataset-specific image input value

def get_dataset_instructions(example_name):
    # Define dataset-specific requirements for image_input
    dataset_requirements = {
        "alarm": False,
        "dog_chair": True,
        "chicken": False,
        "hat": False,
        "lunch_bag": False,
        "mario": False,
        "monkey_chair": True,
        "nutella_new": True,
        "shoe1": False,
        # Add other datasets as needed
    }
    image_input_value = dataset_requirements.get(example_name, True)
    return image_input_value

# Function to display point cloud using Plotly
def display_point_cloud(file_path):
    print(f'[DEBUG] Loading point cloud from: {file_path}')
    point_cloud = trimesh.load(file_path)
    points = np.array(point_cloud.vertices)

    # Check if color data is available
    if hasattr(point_cloud, 'visual') and hasattr(point_cloud.visual, 'vertex_colors'):
        colors = point_cloud.visual.vertex_colors[:, :3] / 255.0  # Normalize RGB to [0, 1]
    else:
        colors = np.array([[0, 0, 1]] * len(points))  # Default to blue if no color info

    # Create a scatter3d plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ))

    print('[DEBUG] Point cloud plot generated.')
    return fig

# Gradio wrapper for the `main` function
def gradio_main(example_name,
    lucid_cam=False, save_video=False
):
    # Set options for the run
    input_path = os.path.join(EXAMPLES_DIR, example_name)  # Set input path based on example name
    opt.data.demo_path = input_path
    opt.single_input = False
    opt.image_data = get_dataset_instructions(example_name)
    opt.crm = False
    opt.lucid_cam = lucid_cam
    opt.save_frames = False
    opt.save_per_view_ply = False
    opt.save_video = save_video
    opt.save_path = 'output/demo'

    pipeline = None
    if opt.single_input:
        from utils.util_demo import setup_crm_diffusion, setup_imagegen_diffusion
        pipeline = setup_crm_diffusion() if opt.crm else setup_imagegen_diffusion()

    # Generate results using the preloaded graph and pipeline
    gen(opt, graph, pipeline=pipeline)

    # Retrieve results (assuming results are saved to opt.output_path/preds)
    output_folder = f"{opt.output_path}/preds"
    video_path = os.path.join(output_folder, f"{opt.output_path.split('/')[-1]}_video_rgb.mp4")
    image_path = os.path.join(output_folder, "2_ccm_input.png")
    point_cloud_path = os.path.join(output_folder, "2_seen_surface_pred.ply")

    # Check if the files exist and return appropriate paths
    video_result = video_path if os.path.exists(video_path) else None
    image_result = image_path if os.path.exists(image_path) else None
    point_cloud_result = point_cloud_path if os.path.exists(point_cloud_path) else None

    return image_result, video_result, point_cloud_result

# Gradio Interface
example_options = list_example_options()

with gr.Blocks() as demo:
    gr.Markdown("# LucidFusion Demo\nRun the 3D generation pipeline with unposed images.")

    # Split the interface into two columns
    with gr.Row():
        with gr.Column(scale=1):
            # Dropdown for selecting example
            example_name = gr.Dropdown(choices=example_options, label="Select Example Dataset", value=example_options[0])
            # Display example images
            example_gallery = gr.Gallery(label="Example Input Images", show_label=True, columns=3, height="400px")

            # Dataset-specific instructions (with larger text)
            
            # Update gallery and instructions when the example dataset changes
            example_name.change(fn=display_example_images, inputs=[example_name], outputs=[example_gallery])
            
            # Add other inputs for the model settings
            lucid_cam = gr.Checkbox(label="Use Lucid Cam", value=True)
            save_video = gr.Checkbox(label="Save Video", value=True)

            # Run button
            run_button = gr.Button("Run Generation")

        with gr.Column(scale=1):
            # Output video
            output_image = gr.Image(label="Generated Image")
            output_video = gr.Video(label="Generated Video")
            output_point_cloud = gr.File(label="Download Point Cloud (PLY)")
    
    # Load initial example images and instructions when the demo starts
    demo.load(fn=display_example_images, inputs=[example_name], outputs=[example_gallery])
    # demo.load(fn=get_dataset_instructions, inputs=[example_name], outputs=[dataset_instructions])

    # Define what happens when the Run button is clicked
    run_button.click(
        fn=gradio_main,
        inputs=[example_name, lucid_cam, save_video],
        outputs=[output_image, output_video, output_point_cloud]
    )

# Run the demo
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9999)