import argparse

from optimum.onnxruntime import ORTConfig, ORTQuantizer

if __name__ == "__main__":

    # usage: python download_data.py --org_name notion --sources appstore,playstore --savepath data --start_batch 10 --end_batch 12 --env staging
    my_parser = argparse.ArgumentParser(description="ONNX Conversion")

    my_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed used to ensure reproducibility across runs",
    )

    my_parser.add_argument(
        "--opt_level",
        type=int,
        default=None,
        help="Optimization level performed by ONNX Runtime of the loaded graph.\
            Supported optimization level are 0, 1, 2 and 99.\
            0 will disable all optimizations.\
            1 will enable basic optimizations.\
            2 will enable basic and extended optimizations, including complex node fusions applied to the nodes\
            assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent.\
            99 will enable all available optimizations including layout optimizations.",
    )

    my_parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="""Whether to optimize the model for GPU inference.
            The optimized graph might contain operators for GPU or CPU only when opt_level > 1""",
    )

    my_parser.add_argument(
        "--only_onnxruntime",
        type=bool,
        default=False,
        help="Whether to only use ONNX Runtime to optimize the model and no graph fusion in Python.",
    )
    my_parser.add_argument(
        "--quantization_approach",
        default="dynamic",
        type=str,
        help="The quantization approach to apply. Supported approach are static and dynamic",
    )

    my_parser.add_argument(
        "--optimize_model",
        default=True,
        type=bool,
        help="Whether to optimize the model before quantization.",
    )

    my_parser.add_argument(
        "--weight_type",
        default="uint8",
        type=str,
        help="The quantization data type of weight. Supported data type are uint8 and int8.",
    )

    my_parser.add_argument("--model_path", type=str, help="Path to the saved model")
    my_parser.add_argument("--output_dir", type=str, help="Path for saving the onnx")

    my_parser.add_argument(
        "--feature", type=str, default="sequence-classification", help="Model task"
    )

    args = my_parser.parse_args()

    opt_level = args.opt_level
    seed = args.seed
    use_gpu = args.use_gpu
    only_onnxruntime = args.only_onnxruntime
    quan_app = args.quantization_approach
    opt_model = args.optimize_model
    weight_type = args.weight_type

    ort_config = ORTConfig(
        opt_level=opt_level,
        seed=seed,
        use_gpu=use_gpu,
        only_onnxruntime=only_onnxruntime,
        quantization_approach=quan_app,
        optimize_model=opt_model,
        weight_type=weight_type,
    )

    ort_quan = ORTQuantizer(ort_config)

    ort_quan.fit(
        model_name_or_path=args.model_path,
        output_dir=args.output_dir,
        feature=args.feature,
    )
