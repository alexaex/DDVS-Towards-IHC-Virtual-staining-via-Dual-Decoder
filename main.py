from opt import opt
import shutil
import os
import logging

class GreenFormatter(logging.Formatter):
    """Custom formatter to add green color to log messages"""
    GREEN = '\033[0;32m'
    RESET = '\033[0m'
    
    def format(self, record):
        # Add green color to the message
        record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

def main():
    # Create a custom formatter with green color
    formatter = GreenFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler and set formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )
    
    logging.info(f"Starting training {opt['model_name']} with opt: {opt}")
    if opt['model_name'] == 'DDStainer':
        logging.info(f"Training {opt['model_name']} started.")
        os.makedirs(opt['save_path'], exist_ok=True)
        shutil.copy('opt.py', os.path.join(opt['save_path'], 'opt.py'))
        from trainer.train_ddstainer import train_ddstainer
        trainer = train_ddstainer(opt, has_discriminator=True)
        trainer.print_model()
        trainer.perf_probing()
        trainer.run()
        logging.info(f"Training {opt['model_name']} finished.")
        
    else:
        raise ValueError(f"Model name {opt['model_name']} not supported")

if __name__ == "__main__":
    main()