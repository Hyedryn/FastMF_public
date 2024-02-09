import fastmf.generation as gen
import sys

if __name__ == "__main__":
    scheme_file = "../data/schemes/scheme-HCPMGH_scheme.txt"
    bvals_file = "../data/schemes/scheme-HCPMGH_bvals.txt"
    dic_file = "../data/dictionaries/dictionary-fixedraddist_scheme-HCPMGH.mat"
    synth_HCP_FixRadDist = gen.Synthetizer(scheme_file, bvals_file, dic_file, task_name="fixraddistHCP")
    num_samples = 10000

    base_path = "../tests/fixraddistHCP/"

    run_id = str(sys.argv[1])

    # Generate training data
    synthStandard = synth_HCP_FixRadDist.generateStandardSet(num_samples, run_id=run_id)
    synthStandard.save(base_path)
