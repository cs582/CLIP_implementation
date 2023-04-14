import unittest
import argparse

from src.tests.image_scraping_unit_tests import ImageScrappingUnitTests
from src.tests.nlp_modules_unit_tests import TransformerRadfordUnitTest
from src.tests.nlp_backbones_unit_tests import BackbonesTextUnitTest
from src.tests.nlp_backbones_unit_tests import BackbonesTextGPUUnitTest
from src.tests.cv_patches_unit_tests import UtilsTest
from src.tests.cv_modules_unit_tests import ResnetModulesUnitTest
from src.tests.cv_backbones_unit_tests import BackbonesUnitTest
from src.tests.cv_backbones_unit_tests import BackbonesUnitTestGPU
from src.tests.nlp_tokenization_unit_tests import TokenizationUnitTest



parser = argparse.ArgumentParser(
    prog='CLIP Unit Tests',
    description='Unit Tests for CLIP Development',
    epilog='Unit Tests used to speed up the development of CLIP and ease error traceback before main training.'
)

parser.add_argument('-heavy', type=bool, default=False, help='Include heavy tasks such as backbones in the unit test.')
parser.add_argument('-use_gpu', type=bool, default=False, help='Test backbones on GPU, heavy must be set to True.')


args = parser.parse_args()

if __name__ == '__main__':
    # Test Image to Patch Transformation for ViT
    utils_test = unittest.TestLoader().loadTestsFromTestCase(UtilsTest)

    # Test ResnetModulesUnitTest
    RN_modules_test = unittest.TestLoader().loadTestsFromTestCase(ResnetModulesUnitTest)

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

    # Tokenization test
    tokenization_test = unittest.TestLoader().loadTestsFromTestCase(TokenizationUnitTest)

    # List all tests to run
    tests_to_run = [utils_test, RN_modules_test, radford_test, image_scrapping_test, tokenization_test]

    if args.heavy:
        print("Stacked Backbones Tests to TestSuite...")
        tests_to_run.append(backbones_cv_test)
        tests_to_run.append(backbones_nlp_test)

        if args.use_gpu:
            print("Stacked Backbones Tests with GPU to TestSuite...")
            tests_to_run.append(backbones_cv_test_gpu)
            tests_to_run.append(backbones_nlp_test_gpu)

    # Test suite that includes all the tests
    suite = unittest.TestSuite(tests_to_run)

    # Run the test suite
    unittest.TextTestRunner().run(suite)