import unittest
import argparse

from src.tests.image_scraping_unit_tests import ImageScrappingUnitTests
from src.tests.nlp_modules_unit_tests import TransformerRadfordUnitTest
from src.tests.nlp_backbones_unit_tests import BackbonesTextUnitTest
from src.tests.nlp_backbones_unit_tests import BackbonesTextGPUUnitTest
from src.tests.nlp_tokenization_unit_tests import TokenizationUnitTest
from src.tests.cv_patches_unit_tests import UtilsTest
from src.tests.cv_modules_unit_tests import ResnetModulesUnitTest
from src.tests.cv_backbones_unit_tests import BackbonesUnitTest
from src.tests.cv_backbones_unit_tests import BackbonesUnitTestGPU
from src.tests.clip_core_unit_tests import CLIPUnitTest
from src.tests.clip_core_unit_tests import CLIPGPUUnitTest



parser = argparse.ArgumentParser(
    prog='CLIP Unit Tests',
    description='Unit Tests for CLIP Development',
    epilog='Unit Tests used to speed up the development of CLIP and ease error traceback before main training.'
)

parser.add_argument('-heavy', type=bool, default=False, help='Include heavy tasks such as backbones in the unit test.')
parser.add_argument('-cpu', type=bool, default=False, help='Test backbones on CPU, heavy must be set to True.')
parser.add_argument('-gpu', type=bool, default=False, help='Test backbones on GPU, heavy must be set to True.')
parser.add_argument('-test_n', type=int, default=0, help='0: All tests. 1: Modules and Backbones. 2: CLIP only.')


args = parser.parse_args()

if __name__ == '__main__':
    # Test Image to Patch Transformation for ViT
    utils_test = unittest.TestLoader().loadTestsFromTestCase(UtilsTest)

    # Test ResnetModulesUnitTest
    rn_modules_test = unittest.TestLoader().loadTestsFromTestCase(ResnetModulesUnitTest)

    # Transformer Radford Unit Test
    radford_test = unittest.TestLoader().loadTestsFromTestCase(TransformerRadfordUnitTest)

    # Test Image Scrapping
    image_scrapping_test = unittest.TestLoader().loadTestsFromTestCase(ImageScrappingUnitTests)

    # Test Backbones
    backbones_cv_test = unittest.TestLoader().loadTestsFromTestCase(BackbonesUnitTest)
    backbones_nlp_test = unittest.TestLoader().loadTestsFromTestCase(BackbonesTextUnitTest)

    # Test Backbones on GPU
    backbones_cv_test_gpu = unittest.TestLoader().loadTestsFromTestCase(BackbonesUnitTestGPU)
    backbones_nlp_test_gpu = unittest.TestLoader().loadTestsFromTestCase(BackbonesTextGPUUnitTest)

    # Test CLIP
    clip_unit_test = unittest.TestLoader().loadTestsFromTestCase(CLIPUnitTest)
    clip_unit_test_gpu = unittest.TestLoader().loadTestsFromTestCase(CLIPGPUUnitTest)

    # Tokenization test
    tokenization_test = unittest.TestLoader().loadTestsFromTestCase(TokenizationUnitTest)

    # List all tests to run
    tests_to_run = []
    if args.test_n in [0, 1]:
        tests_to_run = [utils_test, tokenization_test, rn_modules_test, radford_test, image_scrapping_test]

    if args.heavy:
        if args.cpu and args.test_n in [0, 1]:
            print("Stacked Backbones and CLIP Tests to TestSuite...")
            tests_to_run.append(backbones_cv_test)
            tests_to_run.append(backbones_nlp_test)
        if args.cpu and args.test_n in [0, 2]:
            print("Stacked CLIP Tests to TestSuite...")
            tests_to_run.append(clip_unit_test)

        if args.gpu:
            if args.test_n in [0, 1]:
                print("Stacked Backbones with GPU to TestSuite...")
                tests_to_run.append(backbones_cv_test_gpu)
                tests_to_run.append(backbones_nlp_test_gpu)
            if args.test_n in [0, 2]:
                print("Stacked CLIP Tests with GPU to TestSuite...")
                tests_to_run.append(clip_unit_test_gpu)

    if len(tests_to_run)==0:
        raise ValueError("Choose more than one test to perform.")

    # Test suite that includes all the tests
    suite = unittest.TestSuite(tests_to_run)

    # Run the test suite
    unittest.TextTestRunner().run(suite)