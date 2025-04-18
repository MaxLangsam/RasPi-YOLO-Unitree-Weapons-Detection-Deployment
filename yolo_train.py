from ultralytics import YOLO
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Custom Training Script")
    
    # General model + config
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to dataset YAML")
    parser.add_argument("--init_model", type=str, default="yolov8n.pt", help="Pretrained YOLO model path")
    
    # Training parameters
    parser.add_argument("--name", type=str, default="yolo_run", help="Run name (for saving results)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=0, help="GPU index (set -1 for CPU)")
    parser.add_argument("--image_size", type=str, default="640x640", help="Image size as WxH (e.g. 640x640)")
    parser.add_argument("--resume_training", action='store_true', help="Resume training from last checkpoint")
    
    # Export options
    parser.add_argument("--export_only", action='store_true', help="Only export the model (skip training)")
    parser.add_argument("--export_format", type=str, default="onnx", help="Export format (onnx, imx)")
    parser.add_argument("--export_config", type=str, default=None, help="ONNX export config (optional)")
    parser.add_argument("--int8_weights", action='store_true', help="Export weights in INT8 format")
    
    # Validation
    parser.add_argument("--val_model", action='store_true', help="Only validate the model (no training)")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Detect device
    use_cuda = torch.cuda.is_available()
    device_index = args.device if use_cuda and args.device >= 0 else "cpu"
    print(f"✅ Device in use: {'cuda:' + str(args.device) if use_cuda and args.device >= 0 else 'CPU'}")

    # Parse image size
    image_h, image_w = map(int, args.image_size.split('x'))
    image_size = [image_h, image_w]

    # Load model
    model = YOLO(args.init_model)

    # Train
    if not args.export_only and not args.val_model:
        print("🚀 Starting YOLO training...")
        model.train(
            data=args.config,
            epochs=args.epochs,
            imgsz=image_size,
            save=True,
            device=device_index,
            name=args.name,
            batch=16,
            resume=args.resume_training,
            cache=False,
            project=args.name,
            optimizer='AdamW'  # ✅ Using AdamW optimizer
        )

    # Validate only
    elif args.val_model:
        print("🔍 Validating model...")
        model.val(name=args.name, project=args.name, device=device_index)

    # Export model
    if args.export_format:
        print(f"📦 Exporting model as {args.export_format.upper()}...")
        export_kwargs = {
            "format": args.export_format,
            "int8": args.int8_weights,
            "imgsz": image_size,
            "device": device_index,
            "data": args.export_config,
            "opset": 11,
            "name": args.name,
            "project": args.name
        }

        # Add only if not IMX (not supported there)
        if args.export_format.lower() != "imx":
            export_kwargs["nms"] = True
            export_kwargs["batch"] = 16

        model.export(**export_kwargs)

if __name__ == "__main__":
    main()
