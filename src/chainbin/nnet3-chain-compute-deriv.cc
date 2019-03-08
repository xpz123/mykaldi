// nnet3bin/nnet3-chain-compute-post.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2015   Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "chain/chain-denominator.h"
#include "chain/chain-training.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Compute posteriors from 'denominator FST' of chain model and optionally "
        "map them to phones.\n"
        "\n"
        "Usage: nnet3-chain-compute-deriv [options] <nnet-out-respecifier> <den-fst> <nnet-in> <sup-rspecifier> <deriv-wspecifier>\n"
        " e.g.: nnet3-chain-compute-deriv ark:nnet.out den.fst final.raw ark:sup.txt ark:nnet_prediction.ark\n";
        

    ParseOptions po(usage);
    Timer timer;

    BaseFloat leaky_hmm_coefficient = 0.1;
    NnetSimpleComputationOptions opts;
    opts.acoustic_scale = 1.0; // by default do no acoustic scaling.

    std::string use_gpu = "yes";

    std::string transform_mat_rxfilename;
    std::string ivector_rspecifier,
                online_ivector_rspecifier,
                utt2spk_rspecifier;
    opts.Register(&po);

    
    
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("leaky-hmm-coefficient", &leaky_hmm_coefficient, "'Leaky HMM' "
                "coefficient: smaller values will tend to lead to more "
                "confident posteriors.  0.1 is what we normally use in "
                "training.");
    po.Register("transform-mat", &transform_mat_rxfilename, "Location to read "
                "the matrix to transform posteriors to phones.  Matrix is "
                "of dimension num-phones by num-pdfs.");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

	std::string matrix_rspecifier = po.GetArg(1),
		den_fst_rxfilename = po.GetArg(2),
		nnet_rxfilename = po.GetArg(3),
		sup_rspecifier = po.GetArg(4),
		deriv_wspecifier = po.GetArg(5);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    //CollapseModel(CollapseModelConfig(), &nnet);

    //RandomAccessBaseFloatMatrixReader online_ivector_reader(
    //    online_ivector_rspecifier);
    //RandomAccessBaseFloatVectorReaderMapped ivector_reader(
    //    ivector_rspecifier, utt2spk_rspecifier);

    //CachingOptimizingCompiler compiler(nnet, opts.optimize_config);

    chain::ChainTrainingOptions chain_opts;
    // the only option that actually gets used here is
    // opts_.leaky_hmm_coefficient, and that's the only one we expose on the
    // command line.
    chain_opts.leaky_hmm_coefficient = leaky_hmm_coefficient;

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);
    int32 num_pdfs = nnet.OutputDim("output");
    if (num_pdfs < 0) {
      KALDI_ERR << "Neural net '" << nnet_rxfilename
                << "' has no output named 'output'";
    }
    chain::DenominatorGraph den_graph(den_fst, num_pdfs);

	SequentialBaseFloatMatrixReader nnet_mat_reader(matrix_rspecifier);

	chain::RandomAccessSupervisionReader supervision_reader(
		sup_rspecifier);

    BaseFloatMatrixWriter matrix_writer(deriv_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 tot_input_frames = 0, tot_output_frames = 0;
    double tot_forward_prob = 0.0;



    for (; !nnet_mat_reader.Done(); nnet_mat_reader.Next()) {
      std::string utt = nnet_mat_reader.Key();
      Matrix<BaseFloat> nnet_out (nnet_mat_reader.Value());
      if (nnet_out.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
	  if (!supervision_reader.HasKey(utt)) {
		  KALDI_WARN << "No pdf-level posterior for key " << utt;
		  num_fail++;
		  continue;
	  }
	  CuMatrix<BaseFloat> cu_nnet_out;
	  cu_nnet_out.Swap(&nnet_out);
	  
	  CuMatrix<BaseFloat> cu_deriv_mat(cu_nnet_out.NumRows(), cu_nnet_out.NumCols(), kUndefined);

	  BaseFloat objf, l2_term, weight;

	  chain::ComputeChainObjfAndDeriv(chain_opts, den_graph, supervision_reader.Value(utt), cu_nnet_out, &objf,
		  &l2_term, &weight, &cu_deriv_mat, NULL);

	  if (&cu_deriv_mat != NULL && cu_deriv_mat.NumRows() == cu_nnet_out.NumRows() && cu_deriv_mat.NumCols() == cu_nnet_out.NumCols()) {
		  Matrix<BaseFloat> deriv_mat(cu_deriv_mat);
		  matrix_writer.Write(utt, deriv_mat);
	  }

      

	  KALDI_VLOG(1) << "For utterance " << utt << "objf: " << objf << " weight:" << weight;

      

      num_success++;
      

      
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 input frames/sec is "
              << (elapsed*100.0/tot_input_frames);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    KALDI_LOG << "Overall log-prob per (output) frame was "
              << (tot_forward_prob / tot_output_frames)
              << " over " << tot_output_frames << " frames.";

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
