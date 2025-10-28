from common.utils import get_image_transform
from common.config import cfg


def main():
    # Ensure transform can be created (side-effect: prints nothing, but validates import)
    _ = get_image_transform(is_training=False)
    print("✅ Transform pronto. Seed e utils configurati.")

    # auto-run next
    import section3_model as next_section
    next_section.main()


if __name__ == '__main__':
    main()
