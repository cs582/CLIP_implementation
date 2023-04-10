import unittest
import argparse
from src.tests.cv_patches_unit_tests import UtilsTest
from src.tests.cv_modules_unit_tests import ResnetModulesUnitTest
from src.tests.nlp_unit_tests import TransformerRadfordUnitTest
from src.tests.image_scraping_unit_tests import ImageScrappingUnitTests
from src.tests.cv_modules_unit_tests import BackbonesUnitTest


parser = argparse.ArgumentParser(
    prog='CLIP Unit Tests',
    description='Unit Tests for CLIP Development',
    epilog='Unit Tests used to speed up the development of CLIP and ease error traceback before main training.'
)

parser.add_argument('-exclude_backbones', type=bool, default=True, help='Exclude Backbones from UnitTesting.')

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
    backbones_test = unittest.TestLoader().loadTestsFromTestCase(BackbonesUnitTest)

    # List all tests to run
    tests_to_run = [utils_test, RN_modules_test, radford_test, image_scrapping_test]

    if not args.exclude_backbones:
        tests_to_run.append(backbones_test)

    # Test suite that includes all the tests
    suite = unittest.TestSuite(tests_to_run)

    # Run the test suite
    unittest.TextTestRunner().run(suite)