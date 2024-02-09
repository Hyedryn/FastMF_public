import os
import fastmf.generation as gen

if __name__ == "__main__":
    eval_dir = r"E:\MF"
    scheme_file = os.path.join(eval_dir, "data\\schemes\\scheme-HCPMGH_scheme.txt")
    bvals_file = os.path.join(eval_dir, "data\\schemes\\scheme-HCPMGH_bvals.txt")
    dic_file = os.path.join(eval_dir, "data\\dictionaries\\dictionary-fixedraddist_scheme-HCPMGH.mat")

    print("0. Synthetizer")
    synth_HCP_FixRadDist = gen.Synthetizer(scheme_file, bvals_file, dic_file, task_name="PaperStructuredNoCSF", include_csf=False)
    synth_HCP_FixRadDist.generateStructuredSet()
    base_path = os.path.join(eval_dir, "basedir")

    synthStructured_path = os.path.join(base_path, "synthetizer", "type-structured", "raw",
                                        "type-structured_task-PaperStructuredNoCSF_run-0_raw.pickle")

    print("1. Generator")
    genStructured = gen.Generator(synthStructured_path, base_path, orientation_estimate='MSMTCSD')
    genStructured.computeSphericalHarmonics()
    genStructured.computeNNLSWeights()
    genStructured.computeExhaustiveMF()

    genStructured_2 = gen.Generator(synthStructured_path, base_path, orientation_estimate='GROUNDTRUTH')
    genStructured_2.computeExhaustiveMF()

    print("2. Formatter")
    formatterStructured = gen.DataFormatter(base_path, "PaperStructuredNoCSF", "PaperSmalltest", dic_file,
                                            ["01"], "structured", [7500, 0, 0])

    formatterStructured.genNNLSTarget(include_csf = False)
    formatterStructured.genNNLSInput(include_csf = False)
    formatterStructured.genSphericalHarmonicInput(include_csf = False)
    formatterStructured.genSphericalHarmonicTarget(include_csf = False)
