#include <string>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <exception>
#include <algorithm>
#include <cwctype>
#include <numeric>

#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "online2/online-ivector-feature.h"
#include "lat/lattice-functions.h"
#include "decoder/faster-decoder.h"
#include "base/timer.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		typedef kaldi::int32 int32;
		using fst::SymbolTable;
		using fst::Fst;
		using fst::StdArc;

		const char *usage =
			"Decode(faster) using nnet3 neural net model.\n"
			"Usage: nnet3-decode-faster [options] <nnet-in> <fst-in|fsts-rspecifier> <features-rspecifier>"
			" <word-symbol-table> <words-wspecifier> \n";


		ParseOptions po(usage);

		
		Timer timer;
		NnetSimpleComputationOptions decodable_opts;
		FasterDecoderOptions decoder_opts;

		decoder_opts.Register(&po, false);
		decodable_opts.Register(&po);
		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
			po.PrintUsage();
			exit(1);
		}

		std::string model_in_filename = po.GetArg(1),
			fst_in_str = po.GetArg(2),
			feature_rspecifier = po.GetArg(3),
			word_syms_filename = po.GetArg(4),
			words_wspecifier = po.GetArg(5);

		TokenVectorWriter words_writer(words_wspecifier);

		TransitionModel trans_model;
		AmNnetSimple am_nnet;
		{
			bool binary;
			Input ki(model_in_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_nnet.Read(ki.Stream(), binary);
			SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
			SetDropoutTestMode(true, &(am_nnet.GetNnet()));
			CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
		}

		fst::SymbolTable *word_syms = NULL;
		word_syms = fst::SymbolTable::ReadText(word_syms_filename);

		CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
			decodable_opts.optimize_config);

		Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
		SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		int num_success = 0, num_fail = 0;
		int frame_count = 0;
		{
			FasterDecoder decoder(*decode_fst, decoder_opts);
			for (; !feature_reader.Done(); feature_reader.Next()) {
				std::string utt = feature_reader.Key();
				const Matrix<BaseFloat> &features(feature_reader.Value());
				if (features.NumRows() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					num_fail++;
					continue;
				}
				//Do not support ivector.
				DecodableAmNnetSimple nnet_decodable(
					decodable_opts, trans_model, am_nnet,
					features, NULL, NULL,
					1, &compiler);
				decoder.Decode(&nnet_decodable);
				if (decoder.ReachedFinal()) {
					try {
						fst::VectorFst<LatticeArc> decoded;
						decoder.GetBestPath(&decoded);
						std::vector<int32> alignment;
						std::vector<int32> words;
						LatticeWeight weight;
						GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
						std::vector<std::string> words_txt;
						for (size_t i = 0; i < words.size(); i++) {
							std::string w = word_syms->Find(words[i]);
							words_txt.push_back(w);
						}
						words_writer.Write(utt, words_txt);
						frame_count += nnet_decodable.NumFramesReady();
						num_success++;
					}
					catch (std::exception& e) {
						KALDI_WARN << "Error Message" << e.what() << " for utterance: " << utt;
						num_fail++;
					}
				}
				else {
					num_fail++;
				}
			}
		}

		delete decode_fst;
		delete word_syms;

		kaldi::int64 input_frame_count =
			frame_count * decodable_opts.frame_subsampling_factor;

		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken " << elapsed
			<< "s: real-time factor assuming 100 frames/sec is "
			<< (elapsed * 100.0 / input_frame_count);
		KALDI_LOG << "Done " << num_success << " utterances, failed for "
			<< num_fail;
		if (num_success > 0) return 0;
		else return 1;
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
