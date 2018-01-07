#include <float.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CuTest.h"
#include "../asr_cdp/asr_cdp_lib.h"

float input_spectrogram[] = {
	1.661343774f, 1.895540191f, 1.58082556f,
	1.989696895f, 1.487300705f, 1.41231362f,
	1.396295828f, 1.690109725f, 1.378582209f,
	1.305205604f, 1.090862707f, 1.712547184f,
	1.640117211f, 1.48297118f, 1.529128339f,
	3.30106398f, 5.10531242f, 7.764311597f,
	3.613836417f, 5.663155093f, 7.442975981f,
	10.72116687f, 10.33361021f, 10.68473196f,
	10.5180731f, 10.10881503f, 10.91532679f,
	10.66391731f, 10.8334841f, 10.82630334f,
	7.902745107f, 4.037920361f, 2.505474646f,
	7.06879727f, 4.807981336f, 2.882949863f,
	7.124402818f, 4.209992774f, 2.319300036f,
	7.302486382f, 4.314877236f, 2.170684804f,
	7.151841319f, 4.649747235f, 2.499002642f,
	2.953991295f, 6.541024146f, 3.740091457f,
	1.023471703f, 1.823567133f, 1.354247642f,
	1.267328168f, 1.031676441f, 1.941409353f,
	1.009840476f, 1.426213761f, 1.068586128f,
	1.971877486f, 1.153642186f, 1.079470415f,
	1.51039445f, 1.548338633f, 1.795786212f,
	1.463585363f, 1.828421763f, 1.850302893f
};
int size_of_input_spectrogram = 22;
int word_class_of_input_spectrogram = 2;
int segmentation_of_input_spectrogram[] = {5, 2, 3, 5, 1, 6};
float best_similarity = -0.656212498f;

int feature_vector_size_of_reference = 3;

float reference_silences[] = {
	0.036806757f, 0.432338797f, 0.868043224f,
	1.622042866f, 1.057711219f, 1.802119219f
};
int number_of_reference_silences = 2;

float spectrums_of_references[] = {
	2.546386381f, 4.962106051f, 2.832565004f,
	6.546386381f, 4.962106051f, 2.832565004f,
	2.546386381f, 5.962106051f, 7.832565004f,
	2.822080231f, 2.799094637f, 2.270472804f,
	4.822080231f, 2.799094637f, 5.270472804f,
	3.64694675f, 5.758735131f, 7.392334351f,
	10.64694675f, 10.75873513f, 10.39233435f,
	7.64694675f, 4.758735131f, 2.392334351f,
	2.64694675f, 6.758735131f, 3.392334351f
};

TReferenceItem items_of_first_word[] = {
	{ &spectrums_of_references[0], 1, 2 },
    { &spectrums_of_references[3], 4, 7 },
    { &spectrums_of_references[6], 1, 3 }
};

TReferenceItem items_of_second_word[] = {
	{ &spectrums_of_references[9], 1, 4 },
    { &spectrums_of_references[12], 3, 7 }
};

TReferenceItem items_of_third_word[] = {
	{ &spectrums_of_references[15], 2, 3 },
    { &spectrums_of_references[18], 2, 4 },
    { &spectrums_of_references[21], 4, 6 },
    { &spectrums_of_references[24], 1, 2 }
};

TReference reference_words[] = {
	{ "first", &items_of_first_word[0], 3 },
    { "second", &items_of_second_word[0], 2 },
    { "third", &items_of_third_word[0], 4 }
};
int number_of_reference_words = 3;
TReference *actual_reference_words;
int actual_vocabulary_size, actual_feature_vector_size;
float *actual_reference_silences;
char *references_file_name = "references_for_unit_testing.json";

float _spectrograms_of_first_word[] = {
	0.255667313f, 0.539637326f, 0.850954904f,
	0.683335917f, 0.386432131f, 0.702123754f,
	0.221294416f, 0.2420108f, 0.1683583f,
	0.979095796f, 0.90367873f, 0.03984939f,
	2.546386381f, 4.962106051f, 2.832565004f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.64520901f, 4.969065333f, 2.952172801f,
	6.293224535f, 4.785500195f, 2.652435887f,
	6.163969637f, 4.856817487f, 2.084640886f,
	2.546386381f, 5.962106051f, 7.832565004f,
	0.896959317f, 0.244977909f, 0.40995629f,
	0.373928023f, 0.732905024f, 0.219875699f,
	0.870183298f, 0.076413302f, 0.68748477f,

	0.276788606f, 0.349156841f, 0.634305236f,
	0.173115787f, 0.41732491f, 0.947563373f,
	0.720177321f, 0.229658477f, 0.206851956f,
	0.780297795f, 0.994537145f, 0.413429879f,
	0.971522759f, 0.018322528f, 0.53350812f,
	2.546386381f, 4.962106051f, 2.832565004f,
	2.629772035f, 4.275609783f, 2.387146595f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.926158181f, 4.562178656f, 2.494784741f,
	6.250741217f, 4.08104571f, 2.663454729f,
	6.228642832f, 4.137224215f, 2.003893719f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.574883006f, 4.566329985f, 2.892346538f,
	6.549041395f, 4.697142151f, 2.635172979f,
	2.920918544f, 5.665169927f, 7.307023042f,
	2.546386381f, 5.962106051f, 7.832565004f,
	2.773665215f, 5.601648171f, 7.082660428f,
	0.020871598f, 0.062316637f, 0.4937335f,
	0.604711333f, 0.521355053f, 0.10802153f,
	0.45388677f, 0.840884046f, 0.685368963f,
	0.848853434f, 0.152249906f, 0.637668573f,
	0.775934858f, 0.298009148f, 0.663553611f,

	0.992708162f, 0.854254139f, 0.037595242f,
	0.054360314f, 0.759829238f, 0.346866166f,
	0.31229523f, 0.533856988f, 0.44855794f,
	0.368312648f, 0.544434812f, 0.928400718f,
	0.405998467f, 0.053455511f, 0.239620712f,
	0.331394209f, 0.834553201f, 0.753309918f,
	2.546386381f, 4.962106051f, 2.832565004f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.069175223f, 4.967489018f, 2.957478824f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.74590229f, 4.549970256f, 2.822147466f,
	6.658272823f, 4.851750107f, 2.100323238f,
	2.546386381f, 5.962106051f, 7.832565004f,
	2.772321507f, 5.365383608f, 7.521651901f,
	0.2241955f, 0.569724545f, 0.823643211f,
	0.416141705f, 0.8551779f, 0.404153687f,

	0.496873118f, 0.574424365f, 0.040841935f,
	0.362032631f, 0.398753358f, 0.775320476f,
	2.546386381f, 4.962106051f, 2.832565004f,
	2.859737107f, 4.532267796f, 2.77336938f,
	6.546386381f, 4.962106051f, 2.832565004f,
	6.639145416f, 4.761183912f, 2.762339287f,
	6.161405596f, 4.025144487f, 2.183601296f,
	6.913442117f, 4.883720454f, 2.771270859f,
	6.965589348f, 4.793930677f, 2.501543034f,
	6.546386381f, 4.962106051f, 2.832565004f,
	2.202924059f, 5.387710202f, 7.901096808f,
	2.546386381f, 5.962106051f, 7.832565004f,
	0.197176005f, 0.990053536f, 0.492373015f,
	0.037780486f, 0.49173241f, 0.038260673f,
	0.590479968f, 0.206745391f, 0.94007379f,
	0.349701787f, 0.122898555f, 0.186695282f,
	0.258294537f, 0.763111724f, 0.246679289f,
	0.648366927f, 0.495833272f, 0.203941328f,
	0.45835981f, 0.419455447f, 0.112765931f
};
TSpectrogram spectrograms_of_first_word[] = {
	{ &_spectrograms_of_first_word[0], 13},
    { &_spectrograms_of_first_word[13 * 3], 22 },
    { &_spectrograms_of_first_word[(13 + 22) * 3], 16 },
    { &_spectrograms_of_first_word[(13 + 22 + 16) * 3], 19 }
};

float _spectrograms_of_second_word[] = {
	0.987352563f, 0.253889057f, 0.639706095f,
	0.150855235f, 0.855316977f, 0.410902327f,
	0.232174112f, 0.270577402f, 0.971249879f,
	0.337714989f, 0.729536338f, 0.778514685f,
	0.669519841f, 0.609665352f, 0.595909541f,
	0.767075025f, 0.401909626f, 0.885455452f,
	0.225126455f, 0.42170256f, 0.590929151f,
	2.822080231f, 2.799094637f, 2.270472804f,
	4.263381886f, 2.937886767f, 5.889728541f,
	4.822080231f, 2.799094637f, 5.270472804f,
	4.039947344f, 2.342605281f, 5.566075236f,
	0.900152089f, 0.990071461f, 0.916757066f,
	0.650454523f, 0.671162235f, 0.879839915f,
	0.757773053f, 0.334208546f, 0.635249753f,
	0.162734893f, 0.867472861f, 0.809120134f,
	0.036641691f, 0.031630518f, 0.703191498f,
	0.098383381f, 0.884655528f, 0.538304056f,
	0.562184596f, 0.164693741f, 0.954137996f,
	0.487669579f, 0.869602194f, 0.517920215f,
	0.651159283f, 0.241480573f, 0.960083123f,
	0.25667265f, 0.133972f, 0.284836279f,

	0.854086541f, 0.170622202f, 0.769879639f,
	0.763874043f, 0.600524408f, 0.521084426f,
	2.742244469f, 2.566881679f, 2.050933118f,
	2.822080231f, 2.799094637f, 2.270472804f,
	2.794646652f, 2.167070087f, 2.561865304f,
	2.981318797f, 2.790719117f, 2.646272062f,
	4.99583404f, 2.90262592f, 5.567510032f,
	4.822080231f, 2.799094637f, 5.270472804f,
	4.783264555f, 2.828554009f, 5.201907045f,
	4.412812023f, 2.904360737f, 5.126980983f,
	4.867540254f, 2.004928616f, 5.405508478f,
	4.822080231f, 2.799094637f, 5.270472804f,
	4.080088203f, 2.517750983f, 5.28847124f,
	0.240541937f, 0.077093275f, 0.889866607f,
	0.726637292f, 0.48800009f, 0.893046459f,
	0.248648312f, 0.045802679f, 0.286704401f,

	0.400218549f, 0.44601108f, 0.81319363f,
	0.960333303f, 0.832948218f, 0.537748527f,
	0.409994868f, 0.251094007f, 0.917437985f,
	0.792072071f, 0.221231696f, 0.897176386f,
	0.443522978f, 0.626332477f, 0.097464092f,
	2.822080231f, 2.799094637f, 2.270472804f,
	2.383519832f, 2.866000571f, 2.144020303f,
	4.983136458f, 2.000135491f, 5.313459296f,
	4.419664089f, 2.752869233f, 5.287952202f,
	4.822080231f, 2.799094637f, 5.270472804f,
	4.330414546f, 2.615327724f, 5.657705201f,
	4.045894499f, 2.430095003f, 5.358307315f,
	0.244504489f, 0.824481679f, 0.268847734f,
	0.330022988f, 0.733808449f, 0.367763802f,
	0.027522669f, 0.539105627f, 0.195227231f,
	0.407301872f, 0.548920818f, 0.31856623f,
	0.808629407f, 0.968844041f, 0.154037383f,
	0.643252322f, 0.106976766f, 0.202730982f,

	0.876056718f, 0.999540204f, 0.635071891f,
	0.250619671f, 0.910493207f, 0.031078335f,
	0.912754644f, 0.074805224f, 0.971298517f,
	2.822080231f, 2.799094637f, 2.270472804f,
	2.570608968f, 2.215466147f, 2.754320346f,
	2.822080231f, 2.799094637f, 2.270472804f,
	4.822080231f, 2.799094637f, 5.270472804f,
	4.74124268f, 2.82460103f, 5.908547014f,
	4.192414822f, 2.989837026f, 5.116165805f,
	4.620897285f, 2.630202162f, 5.694567872f,
	4.720798046f, 2.182276963f, 5.11386831f,
	4.23400332f, 2.758734586f, 5.878305091f,
	0.38423093f, 0.555304196f, 0.471639066f,
	0.963094582f, 0.479745358f, 0.572495977f
};
TSpectrogram spectrograms_of_second_word[] = {
	{ &_spectrograms_of_second_word[0], 21 },
    { &_spectrograms_of_second_word[21 * 3], 16 },
    { &_spectrograms_of_second_word[(21 + 16) * 3], 16 },
    { &_spectrograms_of_second_word[(21 + 16 + 18) * 3], 14 }
};

float _spectrograms_of_third_word[] = {
	0.753753803f, 0.07302227f, 0.54187508f,
	0.978702041f, 0.116038173f, 0.946502676f,
	3.988058079f, 5.407641069f, 7.708885515f,
	3.64694675f, 5.758735131f, 7.392334351f,
	10.64694675f, 10.75873513f, 10.39233435f,
	10.19687426f, 10.53924272f, 10.38593319f,
	7.70205775f, 4.77186263f, 2.635599735f,
	7.186918988f, 4.728169816f, 2.744817365f,
	7.346736858f, 4.549035099f, 2.226607181f,
	7.64694675f, 4.758735131f, 2.392334351f,
	2.560479279f, 6.853337074f, 3.950188988f,
	0.509985686f, 0.193033643f, 0.514624273f,
	0.359145994f, 0.007185524f, 0.356938039f,

	0.355101233f, 0.125810876f, 0.5467052f,
	0.206858108f, 0.527445886f, 0.718360949f,
	0.551489493f, 0.292776693f, 0.312453825f,
	3.64694675f, 5.758735131f, 7.392334351f,
	3.472597967f, 5.001843518f, 7.924529163f,
	3.496110833f, 5.81889911f, 7.758541296f,
	10.9881157f, 10.49747245f, 10.69214426f,
	10.64694675f, 10.75873513f, 10.39233435f,
	10.21133604f, 10.77866217f, 10.3942272f,
	10.64694675f, 10.75873513f, 10.39233435f,
	7.009413193f, 4.069417168f, 2.923539323f,
	7.64694675f, 4.758735131f, 2.392334351f,
	7.37248804f, 4.499087048f, 2.975270469f,
	7.64694675f, 4.758735131f, 2.392334351f,
	7.618231429f, 4.820022993f, 2.207227427f,
	7.081633495f, 4.489554913f, 2.969452582f,
	2.64694675f, 6.758735131f, 3.392334351f,
	2.516598996f, 6.712946367f, 3.439039927f,
	0.849166374f, 0.640924505f, 0.833928584f,
	0.009384231f, 0.07261f, 0.63598121f,

	0.877623851f, 0.002163018f, 0.832859284f,
	0.005647602f, 0.71113112f, 0.845948191f,
	0.449425734f, 0.595803563f, 0.272561397f,
	0.199666317f, 0.646223912f, 0.510027483f,
	3.64694675f, 5.758735131f, 7.392334351f,
	3.868094322f, 5.538170389f, 7.563394142f,
	10.17821866f, 10.21542029f, 10.5574385f,
	10.44953826f, 10.14686872f, 10.9172969f,
	10.64694675f, 10.75873513f, 10.39233435f,
	7.94392478f, 4.928779418f, 2.497940043f,
	7.506534762f, 4.132649974f, 2.553005661f,
	7.64694675f, 4.758735131f, 2.392334351f,
	7.6275488f, 4.875435039f, 2.951062283f,
	7.781662775f, 4.094080781f, 2.345514476f,
	2.426708348f, 6.333721371f, 3.45909844f,
	0.793391458f, 0.703106077f, 0.218737817f,
	0.634571503f, 0.449563047f, 0.515100475f,
	0.626234254f, 0.476074711f, 0.891572715f,
	0.581165928f, 0.568877161f, 0.905842522f,
	0.627260975f, 0.058437917f, 0.909389001f,
	0.74452667f, 0.766148531f, 0.719238012f,
	0.389157005f, 0.144653521f, 0.577113283f,

	0.875590245f, 0.147550912f, 0.690694723f,
	0.091599199f, 0.742936924f, 0.790140858f,
	0.054431041f, 0.195188384f, 0.204617611f,
	0.968325183f, 0.257397211f, 0.237661361f,
	0.114653393f, 0.248910292f, 0.354087389f,
	0.388713672f, 0.355608305f, 0.006470561f,
	3.041668682f, 5.696375472f, 7.319749557f,
	3.177766362f, 5.956644092f, 7.247049276f,
	3.64694675f, 5.758735131f, 7.392334351f,
	10.45189226f, 10.3626541f, 10.06887036f,
	10.10004482f, 10.46114867f, 10.38197853f,
	10.64694675f, 10.75873513f, 10.39233435f,
	7.07779371f, 4.150212404f, 2.247327472f,
	7.644861386f, 4.078561261f, 2.895731344f,
	7.016944945f, 4.128976958f, 2.461189576f,
	7.64694675f, 4.758735131f, 2.392334351f,
	7.995076678f, 4.648808563f, 2.510503523f,
	2.610121306f, 6.966993243f, 3.501740084f,
	2.64694675f, 6.758735131f, 3.392334351f,
	0.412133014f, 0.040365624f, 0.361976236f,
	0.636550229f, 0.334599209f, 0.960327558f
};
TSpectrogram spectrograms_of_third_word[] = {
	{ &_spectrograms_of_third_word[0], 13 },
    { &_spectrograms_of_third_word[13 * 3], 20 },
    { &_spectrograms_of_third_word[(13 + 20) * 3], 22 },
    { &_spectrograms_of_third_word[(13 + 20 + 22) * 3], 21 }
};

TTrainDataForWord train_data[] = {
	{ "first", spectrograms_of_first_word, 4 },
	{ "second", spectrograms_of_second_word, 4 },
	{ "third", spectrograms_of_third_word, 4 }
};
int number_of_words_for_training = 3;
int number_of_speech_segments_for_words[] = {3, 2, 4};

float *actual_spectrogram;
int actual_spectrogram_length;
int actual_spectrogram_feature_vector_size;

TTrainDataForWord *actual_train_words;
TTrainDataForWord actual_train_silences;
int actual_number_of_train_words;

char** interesting_words;
int number_of_interesting_words;

void test_calculate_similarity(CuTest *tc)
{
	float spectrogram[] = {
		0.01008f, 0.91680f, 0.85266f,
		0.43764f, 0.85783f, 0.33317f,
		0.16858f, 0.91929f, 0.43628f,
		0.17889f, 0.64864f, 0.58093f,
		0.10396f, 0.78644f, 0.74933f,
		0.43753f, 0.15286f, 0.85823f
	};
	int spectrogram_size = 6, feature_vector_size = 3;
	float reference_spectrum[] = {0.02234f, 0.75742f, 0.69535f};
	float expected = -2.198744217545f;
	float actual = calculate_similarity(spectrogram, spectrogram_size, feature_vector_size,
		reference_spectrum);
	CuAssertDblEquals(tc, expected, actual, 1e-5);
}

void test_find_reference_spectrum(CuTest *tc)
{
	float spectrogram[] = {
		0.01008f, 0.91680f, 0.85266f,
		0.43764f, 0.85783f, 0.33317f,
		0.16858f, 0.91929f, 0.43628f,
		0.17889f, 0.64864f, 0.58093f,
		0.10396f, 0.78644f, 0.74933f,
		0.43753f, 0.15286f, 0.85823f
	};
	int i;
	float tmp_dist_matrix[6 * 6];
	int spectrogram_size = 6, feature_vector_size = 3;
	float expected_reference_spectrum[3] = {0.17889f, 0.64864f, 0.58093f};
	float expected_similarity = -1.993627846127f;
	float actual_reference_spectrum[3];
	float actual_similarity = find_reference_spectrum(spectrogram, spectrogram_size, feature_vector_size,
		actual_reference_spectrum, tmp_dist_matrix);
	CuAssertDblEquals(tc, expected_similarity, actual_similarity, 1e-5);
	for (i = 0; i < feature_vector_size; ++i)
	{
		CuAssertDblEquals(tc, expected_reference_spectrum[i], actual_reference_spectrum[i], 1e-5);
	}
}

void test_recognize_one_sound(CuTest *tc)
{
	float dp_matrix[22 * 6];
	float similarities[3], best_similarities[3];
	int predicted_index = recognize_one_sound(input_spectrogram, size_of_input_spectrogram,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words, number_of_reference_words, best_similarities, similarities, dp_matrix);
	CuAssertIntEquals(tc, word_class_of_input_spectrogram, predicted_index);
	CuAssertDblEquals(tc, best_similarity, best_similarities[word_class_of_input_spectrogram], 1e-5);
}

void test_do_segmentation_01(CuTest *tc)
{
	int i;
	float dp_matrix[22 * 6];
	int lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float actual_similarity = do_segmentation(input_spectrogram, size_of_input_spectrogram,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[2], lengths_of_segments, dp_matrix, dp_matrix_for_lengths);
	CuAssertDblEquals(tc, best_similarity, actual_similarity, 1e-5);
	for (i = 0; i < 6; ++i)
	{
		CuAssertIntEquals(tc, segmentation_of_input_spectrogram[i], lengths_of_segments[i]);
	}
}

void test_do_segmentation_02(CuTest *tc)
{
	int i;
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 4, 1, 4, 1, 3 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_segmentation(train_data[0].spectrograms[0].spectrogram, train_data[0].spectrograms[0].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0], actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_segmentation_03(CuTest *tc)
{
	int i;
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 5, 2, 7, 3, 5 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_segmentation(train_data[0].spectrograms[1].spectrogram, train_data[0].spectrograms[1].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0], actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_segmentation_04(CuTest *tc)
{
	int i;
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 6, 1, 5, 2, 2 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_segmentation(train_data[0].spectrograms[2].spectrogram, train_data[0].spectrograms[2].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0], actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_segmentation_05(CuTest *tc)
{
	int i;
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 2, 2, 6, 2, 7 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_segmentation(train_data[0].spectrograms[3].spectrogram, train_data[0].spectrograms[3].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0], actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_self_segmentation_01(CuTest *tc)
{
	int i;
	float tmp_dist_matrix[22 * 22];
	float dp_matrix[22 * 6];
	int lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float expected_similarity = -0.53366988f;
	float actual_similarity = do_self_segmentation(input_spectrogram, size_of_input_spectrogram,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[2].n, lengths_of_segments, dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
	CuAssertDblEquals(tc, expected_similarity, actual_similarity, 1e-5);
	for (i = 0; i < 6; ++i)
	{
		CuAssertIntEquals(tc, segmentation_of_input_spectrogram[i], lengths_of_segments[i]);
	}
}

void test_do_self_segmentation_02(CuTest *tc)
{
	int i;
	float tmp_dist_matrix[22 * 22];
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 4, 1, 4, 1, 3 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_self_segmentation(train_data[0].spectrograms[0].spectrogram, train_data[0].spectrograms[0].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0].n, actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_self_segmentation_03(CuTest *tc)
{
	int i;
	float tmp_dist_matrix[22 * 22];
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 5, 2, 7, 3, 5 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_self_segmentation(train_data[0].spectrograms[1].spectrogram, train_data[0].spectrograms[1].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0].n, actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_self_segmentation_04(CuTest *tc)
{
	int i;
	float tmp_dist_matrix[22 * 22];
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 6, 1, 5, 2, 2 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_self_segmentation(train_data[0].spectrograms[2].spectrogram, train_data[0].spectrograms[2].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0].n, actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_do_self_segmentation_05(CuTest *tc)
{
	int i;
	float tmp_dist_matrix[22 * 22];
	float dp_matrix[22 * 6];
	int expected_length_of_segments[] = { 2, 2, 6, 2, 7 };
	int actual_lengths_of_segments[6];
	int dp_matrix_for_lengths[22 * 6];
	float similarity = do_self_segmentation(train_data[0].spectrograms[3].spectrogram, train_data[0].spectrograms[3].n,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences,
		reference_words[0].n, actual_lengths_of_segments, dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
	CuAssertTrue(tc, similarity > (-FLT_MAX / 2.0));
	CuAssertTrue(tc, similarity <= 0.0);
	for (i = 0; i < (reference_words[0].n + 2); ++i)
	{
		CuAssertIntEquals(tc, expected_length_of_segments[i], actual_lengths_of_segments[i]);
	}
}

void test_compare_segmentation_01(CuTest *tc)
{
	int segmentation1[] = {
		4, 1, 4, 1, 3,
		5, 2, 7, 3, 5,
		6, 1, 5, 2, 2,
		2, 2, 6, 2, 7,
		7, 1, 3, 10,
		2, 4, 7, 3,
		5, 2, 5, 6,
		3, 3, 6, 2,
		2, 2, 2, 4, 1, 2,
		3, 3, 4, 6, 2, 2,
		4, 2, 3, 5, 1, 7,
		6, 3, 3, 5, 2, 2
	};
	int segmentation2[] = {
		4, 1, 4, 1, 3,
		5, 2, 7, 3, 5,
		6, 1, 5, 2, 2,
		2, 2, 6, 2, 7,
		7, 1, 3, 10,
		2, 4, 7, 3,
		5, 2, 5, 6,
		3, 3, 6, 2,
		2, 2, 2, 4, 1, 2,
		3, 3, 4, 6, 2, 2,
		4, 2, 3, 5, 1, 7,
		6, 3, 3, 5, 2, 2
	};
	int expected = 0;
	int actual = compare_segmentation(segmentation1, segmentation2,
		train_data, number_of_speech_segments_for_words, number_of_words_for_training);
	CuAssertIntEquals(tc, expected, actual);
}

void test_compare_segmentation_02(CuTest *tc)
{
	int segmentation1[] = {
		4, 1, 4, 1, 3,
		5, 2, 7, 3, 5,
		6, 1, 5, 2, 2,
		2, 2, 6, 2, 7,
		7, 1, 3, 10,
		2, 4, 7, 3,
		5, 2, 5, 6,
		3, 3, 6, 2,
		2, 2, 2, 4, 1, 2,
		3, 3, 4, 6, 2, 2,
		4, 2, 3, 5, 1, 7,
		6, 3, 3, 5, 2, 2
	};
	int segmentation2[] = {
		3, 1, 5, 1, 3,
		5, 2, 7, 3, 5,
		6, 1, 5, 2, 2,
		2, 2, 6, 2, 7,
		7, 1, 3, 10,
		3, 4, 5, 4,
		5, 2, 5, 6,
		3, 3, 6, 2,
		1, 1, 2, 7, 1, 1,
		3, 3, 4, 6, 2, 2,
		4, 2, 3, 5, 1, 7,
		6, 3, 3, 5, 2, 2
	};
	int expected = 3;
	int actual = compare_segmentation(segmentation1, segmentation2,
		train_data, number_of_speech_segments_for_words, number_of_words_for_training);
	CuAssertIntEquals(tc, expected, actual);
}

void test_create_references_for_words(CuTest *tc)
{
	int i, j, k;
	if (actual_reference_words != NULL)
	{
		finalize_references(actual_reference_words, actual_vocabulary_size);
		actual_reference_words = NULL;
	}
	actual_vocabulary_size = number_of_words_for_training;
	actual_feature_vector_size = feature_vector_size_of_reference;
	actual_reference_words = create_references_for_words(train_data, number_of_speech_segments_for_words,
		number_of_words_for_training, feature_vector_size_of_reference,
		reference_silences, number_of_reference_silences, 10);
	CuAssertPtrNotNull(tc, actual_reference_words);
	for (i = 0; i < number_of_reference_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_reference_words[i].reference);
		CuAssertIntEquals(tc, reference_words[i].n, actual_reference_words[i].n);
		CuAssertStrEquals(tc, reference_words[i].wordname, actual_reference_words[i].wordname);
		for (j = 0; j < reference_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_reference_words[i].reference[j].spectrum);
			CuAssertIntEquals(tc, reference_words[i].reference[j].m, actual_reference_words[i].reference[j].m);
			CuAssertIntEquals(tc, reference_words[i].reference[j].M, actual_reference_words[i].reference[j].M);
			for (k = 0; k < feature_vector_size_of_reference; ++k)
			{
				CuAssertDblEquals(tc, reference_words[i].reference[j].spectrum[k],
					actual_reference_words[i].reference[j].spectrum[k], 1e-5);
			}
		}
	}
}

void test_calculate_states_number_for_word(CuTest *tc)
{
	char* src_word = "hello";
	int expected = 17;
	int actual = calculate_states_number_for_word(src_word);
	CuAssertIntEquals(tc, expected, actual);
}

void test_join_and_prepare_filename_01(CuTest *tc)
{
	char* basedir = "basedir";
	char filename[1024];
	filename[0] = 't';
	filename[1] = 'e';
	filename[2] = 's';
	filename[3] = 't';
	filename[4] = '.';
	filename[5] = 'f';
	filename[6] = 'b';
	filename[7] = 'a';
	filename[8] = 'n';
	filename[9] = 'k';
	filename[10] = 's';
	filename[11] = 0;
#ifdef _WIN32
	char* expected = "basedir\\test.fbanks.bin";
#else
	char* expected = "basedir/test.fbanks.bin";
#endif
	char* actual = join_and_prepare_filename(basedir, filename);
	CuAssertStrEquals(tc, expected, actual);
}

void test_join_and_prepare_filename_02(CuTest *tc)
{
	char* basedir = "basedir/";
	char filename[1024];
	filename[0] = 't';
	filename[1] = 'e';
	filename[2] = 's';
	filename[3] = 't';
	filename[4] = '.';
	filename[5] = 'f';
	filename[6] = 'b';
	filename[7] = 'a';
	filename[8] = 'n';
	filename[9] = 'k';
	filename[10] = 's';
	filename[11] = 0;
#ifdef _WIN32
	char* expected = "basedir\\test.fbanks.bin";
#else
	char* expected = "basedir/test.fbanks.bin";
#endif
	char* actual = join_and_prepare_filename(basedir, filename);
	CuAssertStrEquals(tc, expected, actual);
}

void test_join_and_prepare_filename_03(CuTest *tc)
{
	char* basedir = "basedir/";
	char filename[1024];
	filename[0] = 's';
	filename[1] = 'u';
	filename[2] = 'b';
	filename[3] = 'd';
	filename[4] = 'i';
	filename[5] = 'r';
	filename[6] = '\\';
	filename[7] = 't';
	filename[8] = 'e';
	filename[9] = 's';
	filename[10] = 't';
	filename[11] = '.';
	filename[12] = 'f';
	filename[13] = 'b';
	filename[14] = 'a';
	filename[15] = 'n';
	filename[16] = 'k';
	filename[17] = 's';
	filename[18] = 0;
#ifdef _WIN32
	char* expected = "basedir\\subdir\\test.fbanks.bin";
#else
	char* expected = "basedir/subdir/test.fbanks.bin";
#endif
	char* actual = join_and_prepare_filename(basedir, filename);
	CuAssertStrEquals(tc, expected, actual);
}

void test_io_references(CuTest *tc)
{
	int ok;
	int i, j, k;
	int actual_silences_number;
	if (actual_reference_words != NULL)
	{
		finalize_references(actual_reference_words, actual_vocabulary_size);
		actual_reference_words = NULL;
	}
	if (actual_reference_silences != NULL)
	{
		free(actual_reference_silences);
		actual_reference_silences = NULL;
	}
	ok = save_references(references_file_name, reference_words, number_of_reference_words,
		feature_vector_size_of_reference, reference_silences, number_of_reference_silences);
	CuAssertTrue(tc, ok);
	ok = load_references(references_file_name, &actual_reference_words, &actual_vocabulary_size,
		&actual_feature_vector_size, &actual_reference_silences, &actual_silences_number);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, feature_vector_size_of_reference, actual_feature_vector_size);
	CuAssertIntEquals(tc, number_of_reference_words, actual_vocabulary_size);
	CuAssertIntEquals(tc, number_of_reference_silences, actual_silences_number);
	CuAssertPtrNotNull(tc, actual_reference_silences);
	for (i = 0; i < (number_of_reference_silences * feature_vector_size_of_reference); ++i)
	{
		CuAssertDblEquals(tc, reference_silences[i], actual_reference_silences[i], 1e-5);
	}
	CuAssertPtrNotNull(tc, actual_reference_words);
	for (i = 0; i < number_of_reference_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_reference_words[i].reference);
		CuAssertIntEquals(tc, reference_words[i].n, actual_reference_words[i].n);
		CuAssertStrEquals(tc, reference_words[i].wordname, actual_reference_words[i].wordname);
		for (j = 0; j < reference_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_reference_words[i].reference[j].spectrum);
			CuAssertIntEquals(tc, reference_words[i].reference[j].m, actual_reference_words[i].reference[j].m);
			CuAssertIntEquals(tc, reference_words[i].reference[j].M, actual_reference_words[i].reference[j].M);
			for (k = 0; k < feature_vector_size_of_reference; ++k)
			{
				CuAssertDblEquals(tc, reference_words[i].reference[j].spectrum[k],
					actual_reference_words[i].reference[j].spectrum[k], 1e-5);
			}
		}
	}
}

void test_load_spectrogram_01(CuTest *tc)
{
#ifdef _WIN32
	char *filename = "testdata\\first\\first01.fbanks.bin";
#else
	char *filename = "testdata/first/first01.fbanks.bin";
#endif
	int ok, i;
	int expected_spec_size = 18;
	int expected_ft_size = 3;
	float expected_data[] = {
		4.83334690e-01f, 7.06965208e-01f, 2.98463136e-01f,
		1.56471357e-01f, 8.97346735e-02f, 1.85559675e-01f,
		9.28982913e-01f, 1.88486159e-01f, 9.67477202e-01f,
		7.73426294e-01f, 7.51656055e-01f, 1.26902061e-03f,
		5.52953660e-01f, 5.79978190e-02f, 3.80673558e-01f,
		2.46011806e+00f, 4.56348848e+00f, 2.22063541e+00f,
		6.35453129e+00f, 4.49206018e+00f, 2.36474371e+00f,
		6.60829878e+00f, 4.86807489e+00f, 2.63377619e+00f,
		6.48463774e+00f, 4.69062185e+00f, 2.33264613e+00f,
		6.94285250e+00f, 4.97646761e+00f, 2.71088958e+00f,
		6.18254185e+00f, 4.75591040e+00f, 2.60909343e+00f,
		6.51077890e+00f, 5.31049967e+00f, 7.01891232e+00f,
		4.09740835e-01f, 5.39178193e-01f, 3.01938832e-01f,
		5.36633134e-02f, 8.90397787e-01f, 4.01526749e-01f,
		6.29373252e-01f, 9.54982340e-01f, 4.84906077e-01f,
		9.61764812e-01f, 1.64244115e-01f, 2.22531587e-01f,
		2.63350576e-01f, 1.53118849e-01f, 3.84879202e-01f,
		6.74470723e-01f, 3.75316322e-01f, 6.37635767e-01f
	};
	if (actual_spectrogram != NULL)
	{
		free(actual_spectrogram);
		actual_spectrogram = NULL;
	}
	ok = load_spectrogram(filename, &actual_spectrogram, &actual_spectrogram_length,
		&actual_spectrogram_feature_vector_size);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, expected_spec_size, actual_spectrogram_length);
	CuAssertIntEquals(tc, expected_ft_size, actual_spectrogram_feature_vector_size);
	for (i = 0; i < (expected_ft_size * expected_spec_size); ++i)
	{
		CuAssertDblEquals(tc, expected_data[i], actual_spectrogram[i], 1e-5);
	}
}

void test_load_spectrogram_02(CuTest *tc)
{
	char *filename = "94e6864f_nohash_0.wav.fbanks.bin";
	int ok, i;
	int expected_spec_size = 98;
	int expected_ft_size = 32;
	if (actual_spectrogram != NULL)
	{
		free(actual_spectrogram);
		actual_spectrogram = NULL;
	}
	ok = load_spectrogram(filename, &actual_spectrogram, &actual_spectrogram_length,
		&actual_spectrogram_feature_vector_size);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, expected_spec_size, actual_spectrogram_length);
	CuAssertIntEquals(tc, expected_ft_size, actual_spectrogram_feature_vector_size);
	for (i = 0; i < (expected_ft_size * expected_spec_size); ++i)
	{
		CuAssertTrue(tc, actual_spectrogram[i] >= 0.0);
	}
}

void test_find_word_01(CuTest *tc)
{
	char *interesting_words[] = { "first", "second", "third" };
	int expected = 1;
	int actual = find_word("second", interesting_words, 3);
	CuAssertIntEquals(tc, expected, actual);
}

void test_find_word_02(CuTest *tc)
{
	char *interesting_words[] = { "first", "second", "third" };
	int expected = -1;
	int actual = find_word("fourth", interesting_words, 3);
	CuAssertIntEquals(tc, expected, actual);
}

void test_load_train_data_01(CuTest *tc)
{
	int i, j, k;
	float cur, prev;
	char *basedir = "testdata";
	char *filename = "distribution_test.json";
	char *interesting_words[] = { "first", "second", "third" };
	int used_words[3];
	int actual_ft_size;
	int ok;
	used_words[0] = 0;
	used_words[1] = 0;
	used_words[2] = 0;
	if (actual_train_words != NULL)
	{
		finalize_train_data(actual_train_words, actual_number_of_train_words);
		actual_train_words = NULL;
		actual_number_of_train_words = 0;
	}
	finalize_train_data_for_word(actual_train_silences);
	ok = load_train_data(filename, basedir, "train", NULL, 0, &actual_ft_size, &actual_train_words,
		&actual_number_of_train_words, &actual_train_silences);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, feature_vector_size_of_reference, actual_ft_size);
	CuAssertIntEquals(tc, number_of_words_for_training, actual_number_of_train_words);
	CuAssertIntEquals(tc, 3, actual_train_silences.n);
	CuAssertTrue(tc, actual_train_silences.wordname == NULL);
	CuAssertPtrNotNull(tc, actual_train_silences.spectrograms);
	for (i = 0; i < actual_train_silences.n; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_silences.spectrograms[i].spectrogram);
		CuAssertTrue(tc, actual_train_silences.spectrograms[i].n > 0);
		prev = -FLT_MAX;
		for (j = 0; j < (actual_ft_size * actual_train_silences.spectrograms[i].n); ++j)
		{
			cur = actual_train_silences.spectrograms[i].spectrogram[j];
			CuAssertTrue(tc, cur >= 0.0);
			CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
			prev = cur;
		}
	}
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_words[i].wordname);
		CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms);
		CuAssertIntEquals(tc, 3, actual_train_words[i].n);
		j = find_word(actual_train_words[i].wordname, interesting_words, 3);
		CuAssertTrue(tc, j >= 0);
		used_words[j] += 1;
	}
	CuAssertIntEquals(tc, 1, used_words[0]);
	CuAssertIntEquals(tc, 1, used_words[1]);
	CuAssertIntEquals(tc, 1, used_words[2]);
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		for (j = 0; j < actual_train_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms[j].spectrogram);
			CuAssertTrue(tc, actual_train_words[i].spectrograms[j].n > 0);
			prev = -FLT_MAX;
			for (k = 0; k < (actual_ft_size * actual_train_words[i].spectrograms[j].n); ++k)
			{
				cur = actual_train_words[i].spectrograms[j].spectrogram[k];
				CuAssertTrue(tc, cur >= 0.0);
				CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
				prev = cur;
			}
		}
	}
}

void test_load_train_data_02(CuTest *tc)
{
	int i, j, k;
	float cur, prev;
	char *basedir = "testdata";
	char *filename = "distribution_test.json";
	char *interesting_words[] = { "first", "second", "third" };
	int used_words[3];
	int actual_ft_size;
	int ok;
	used_words[0] = 0;
	used_words[1] = 0;
	used_words[2] = 0;
	if (actual_train_words != NULL)
	{
		finalize_train_data(actual_train_words, actual_number_of_train_words);
		actual_train_words = NULL;
		actual_number_of_train_words = 0;
	}
	finalize_train_data_for_word(actual_train_silences);
	ok = load_train_data(filename, basedir, "validation", NULL, 0, &actual_ft_size, &actual_train_words,
		&actual_number_of_train_words, &actual_train_silences);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, feature_vector_size_of_reference, actual_ft_size);
	CuAssertIntEquals(tc, number_of_words_for_training, actual_number_of_train_words);
	CuAssertIntEquals(tc, 2, actual_train_silences.n);
	CuAssertTrue(tc, actual_train_silences.wordname == NULL);
	CuAssertPtrNotNull(tc, actual_train_silences.spectrograms);
	for (i = 0; i < actual_train_silences.n; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_silences.spectrograms[i].spectrogram);
		CuAssertTrue(tc, actual_train_silences.spectrograms[i].n > 0);
		prev = -FLT_MAX;
		for (j = 0; j < (actual_ft_size * actual_train_silences.spectrograms[i].n); ++j)
		{
			cur = actual_train_silences.spectrograms[i].spectrogram[j];
			CuAssertTrue(tc, cur >= 0.0);
			CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
			prev = cur;
		}
	}
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_words[i].wordname);
		CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms);
		CuAssertIntEquals(tc, 2, actual_train_words[i].n);
		j = find_word(actual_train_words[i].wordname, interesting_words, 3);
		CuAssertTrue(tc, j >= 0);
		used_words[j] += 1;
	}
	CuAssertIntEquals(tc, 1, used_words[0]);
	CuAssertIntEquals(tc, 1, used_words[1]);
	CuAssertIntEquals(tc, 1, used_words[2]);
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		for (j = 0; j < actual_train_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms[j].spectrogram);
			CuAssertTrue(tc, actual_train_words[i].spectrograms[j].n > 0);
			prev = -FLT_MAX;
			for (k = 0; k < (actual_ft_size * actual_train_words[i].spectrograms[j].n); ++k)
			{
				cur = actual_train_words[i].spectrograms[j].spectrogram[k];
				CuAssertTrue(tc, cur >= 0.0);
				CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
				prev = cur;
			}
		}
	}
}

void test_load_train_data_03(CuTest *tc)
{
	int i, j, k;
	float cur, prev;
	char *basedir = "testdata";
	char *filename = "distribution_test.json";
	char *interesting_words[] = { "first", "second", "third" };
	int used_words[3];
	int actual_ft_size;
	int ok;
	used_words[0] = 0;
	used_words[1] = 0;
	used_words[2] = 0;
	if (actual_train_words != NULL)
	{
		finalize_train_data(actual_train_words, actual_number_of_train_words);
		actual_train_words = NULL;
		actual_number_of_train_words = 0;
	}
	finalize_train_data_for_word(actual_train_silences);
	ok = load_train_data(filename, basedir, "test", NULL, 0, &actual_ft_size, &actual_train_words,
		&actual_number_of_train_words, &actual_train_silences);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, feature_vector_size_of_reference, actual_ft_size);
	CuAssertIntEquals(tc, number_of_words_for_training, actual_number_of_train_words);
	CuAssertIntEquals(tc, 1, actual_train_silences.n);
	CuAssertTrue(tc, actual_train_silences.wordname == NULL);
	CuAssertPtrNotNull(tc, actual_train_silences.spectrograms);
	for (i = 0; i < actual_train_silences.n; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_silences.spectrograms[i].spectrogram);
		CuAssertTrue(tc, actual_train_silences.spectrograms[i].n > 0);
		prev = -FLT_MAX;
		for (j = 0; j < (actual_ft_size * actual_train_silences.spectrograms[i].n); ++j)
		{
			cur = actual_train_silences.spectrograms[i].spectrogram[j];
			CuAssertTrue(tc, cur >= 0.0);
			CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
			prev = cur;
		}
	}
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_words[i].wordname);
		CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms);
		CuAssertIntEquals(tc, 1, actual_train_words[i].n);
		j = find_word(actual_train_words[i].wordname, interesting_words, 3);
		CuAssertTrue(tc, j >= 0);
		used_words[j] += 1;
	}
	CuAssertIntEquals(tc, 1, used_words[0]);
	CuAssertIntEquals(tc, 1, used_words[1]);
	CuAssertIntEquals(tc, 1, used_words[2]);
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		for (j = 0; j < actual_train_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms[j].spectrogram);
			CuAssertTrue(tc, actual_train_words[i].spectrograms[j].n > 0);
			prev = -FLT_MAX;
			for (k = 0; k < (actual_ft_size * actual_train_words[i].spectrograms[j].n); ++k)
			{
				cur = actual_train_words[i].spectrograms[j].spectrogram[k];
				CuAssertTrue(tc, cur >= 0.0);
				CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
				prev = cur;
			}
		}
	}
}

void test_load_train_data_04(CuTest *tc)
{
	int i, j, k;
	float cur, prev;
	char *basedir = "testdata";
	char *filename = "distribution_test.json";
	char *interesting_words[] = { "first", "second" };
	int used_words[2];
	int actual_ft_size;
	int ok;
	used_words[0] = 0;
	used_words[1] = 0;
	if (actual_train_words != NULL)
	{
		finalize_train_data(actual_train_words, actual_number_of_train_words);
		actual_train_words = NULL;
		actual_number_of_train_words = 0;
	}
	finalize_train_data_for_word(actual_train_silences);
	ok = load_train_data(filename, basedir, "train", interesting_words, 2, &actual_ft_size, &actual_train_words,
		&actual_number_of_train_words, &actual_train_silences);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, feature_vector_size_of_reference, actual_ft_size);
	CuAssertIntEquals(tc, 2, actual_number_of_train_words);
	CuAssertIntEquals(tc, 3, actual_train_silences.n);
	CuAssertTrue(tc, actual_train_silences.wordname == NULL);
	CuAssertPtrNotNull(tc, actual_train_silences.spectrograms);
	for (i = 0; i < actual_train_silences.n; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_silences.spectrograms[i].spectrogram);
		CuAssertTrue(tc, actual_train_silences.spectrograms[i].n > 0);
		prev = -FLT_MAX;
		for (j = 0; j < (actual_ft_size * actual_train_silences.spectrograms[i].n); ++j)
		{
			cur = actual_train_silences.spectrograms[i].spectrogram[j];
			CuAssertTrue(tc, cur >= 0.0);
			CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
			prev = cur;
		}
	}
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		CuAssertPtrNotNull(tc, actual_train_words[i].wordname);
		CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms);
		CuAssertIntEquals(tc, 3, actual_train_words[i].n);
		j = find_word(actual_train_words[i].wordname, interesting_words, 2);
		CuAssertTrue(tc, j >= 0);
		used_words[j] += 1;
	}
	CuAssertIntEquals(tc, 1, used_words[0]);
	CuAssertIntEquals(tc, 1, used_words[1]);
	for (i = 0; i < actual_number_of_train_words; ++i)
	{
		for (j = 0; j < actual_train_words[i].n; ++j)
		{
			CuAssertPtrNotNull(tc, actual_train_words[i].spectrograms[j].spectrogram);
			CuAssertTrue(tc, actual_train_words[i].spectrograms[j].n > 0);
			prev = -FLT_MAX;
			for (k = 0; k < (actual_ft_size * actual_train_words[i].spectrograms[j].n); ++k)
			{
				cur = actual_train_words[i].spectrograms[j].spectrogram[k];
				CuAssertTrue(tc, cur >= 0.0);
				CuAssertTrue(tc, fabs(cur - prev) > 1e-5);
				prev = cur;
			}
		}
	}
}

void test_strip_line_01(CuTest *tc)
{
	char buffer[1024];
	memset(buffer, 0, 1024 * sizeof(char));
	strcpy(buffer, "Hello, world!");
	CuAssertStrEquals(tc, "Hello, world!", strip_line(buffer));
}

void test_strip_line_02(CuTest *tc)
{
	char buffer[1024];
	memset(buffer, 0, 1024 * sizeof(char));
	strcpy(buffer, "\tHello, world! \n");
	CuAssertStrEquals(tc, "Hello, world!", strip_line(buffer));
}

void test_strip_line_03(CuTest *tc)
{
	char buffer[1024];
	memset(buffer, 0, 1024 * sizeof(char));
	strcpy(buffer, "                                 Hello, world! \r\n");
	CuAssertStrEquals(tc, "Hello, world!", strip_line(buffer));
}

void test_load_interesting_words(CuTest *tc)
{
	int i, ok;
	char *expected[] = {
		"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
	};
	int expected_n = 10;
	if (interesting_words != NULL)
	{
		finalize_interesting_words(interesting_words, number_of_interesting_words);
		interesting_words = NULL;
		number_of_interesting_words = 0;
	}
	ok = load_interesting_words("interesting_words_test.txt", &interesting_words, &number_of_interesting_words);
	CuAssertTrue(tc, ok);
	CuAssertIntEquals(tc, expected_n, number_of_interesting_words);
	for (i = 0; i < expected_n; ++i)
	{
		CuAssertStrEquals(tc, expected[i], interesting_words[i]);
	}
}

CuSuite* ASR_CDP_GetSuite()
{
	CuSuite* suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, test_calculate_similarity);
	SUITE_ADD_TEST(suite, test_find_reference_spectrum);
	SUITE_ADD_TEST(suite, test_recognize_one_sound);
	SUITE_ADD_TEST(suite, test_do_segmentation_01);
	SUITE_ADD_TEST(suite, test_do_segmentation_02);
	SUITE_ADD_TEST(suite, test_do_segmentation_03);
	SUITE_ADD_TEST(suite, test_do_segmentation_04);
	SUITE_ADD_TEST(suite, test_do_segmentation_05);
	SUITE_ADD_TEST(suite, test_do_self_segmentation_01);
	SUITE_ADD_TEST(suite, test_do_self_segmentation_02);
	SUITE_ADD_TEST(suite, test_do_self_segmentation_03);
	SUITE_ADD_TEST(suite, test_do_self_segmentation_04);
	SUITE_ADD_TEST(suite, test_do_self_segmentation_05);
	SUITE_ADD_TEST(suite, test_compare_segmentation_01);
	SUITE_ADD_TEST(suite, test_compare_segmentation_02);
	SUITE_ADD_TEST(suite, test_create_references_for_words);
	SUITE_ADD_TEST(suite, test_calculate_states_number_for_word);
	SUITE_ADD_TEST(suite, test_join_and_prepare_filename_01);
	SUITE_ADD_TEST(suite, test_join_and_prepare_filename_02);
	SUITE_ADD_TEST(suite, test_join_and_prepare_filename_03);
	SUITE_ADD_TEST(suite, test_io_references);
	SUITE_ADD_TEST(suite, test_load_spectrogram_01);
	SUITE_ADD_TEST(suite, test_load_spectrogram_02);
	SUITE_ADD_TEST(suite, test_find_word_01);
	SUITE_ADD_TEST(suite, test_find_word_02);
	SUITE_ADD_TEST(suite, test_load_train_data_01);
	SUITE_ADD_TEST(suite, test_load_train_data_02);
	SUITE_ADD_TEST(suite, test_load_train_data_03);
	SUITE_ADD_TEST(suite, test_load_train_data_04);
	SUITE_ADD_TEST(suite, test_strip_line_01);
	SUITE_ADD_TEST(suite, test_strip_line_02);
	SUITE_ADD_TEST(suite, test_strip_line_03);
	SUITE_ADD_TEST(suite, test_load_interesting_words);
	return suite;
}

void RunAllTests(void)
{
	CuString *output = CuStringNew();
	CuSuite* suite = CuSuiteNew();

	CuSuiteAddSuite(suite, ASR_CDP_GetSuite());

	CuSuiteRun(suite);
	CuSuiteSummary(suite, output);
	CuSuiteDetails(suite, output);
	printf("%s\n", output->buffer);
}

int main(void)
{
	interesting_words = NULL;
	number_of_interesting_words = 0;
	actual_reference_words = NULL;
	actual_reference_silences = NULL;
	actual_vocabulary_size = 0;
	actual_feature_vector_size = 0;
	actual_spectrogram_length = 0;
	actual_spectrogram_feature_vector_size = 0;
	actual_spectrogram = NULL;
	actual_train_words = NULL;
	actual_train_silences.wordname = NULL;
	actual_train_silences.spectrograms = NULL;
	actual_train_silences.n = 0;
	actual_number_of_train_words = 0;
	RunAllTests();
	if (actual_reference_words != NULL)
	{
		finalize_references(actual_reference_words, actual_vocabulary_size);
		actual_reference_words = NULL;
	}
	if (actual_reference_silences != NULL)
	{
		free(actual_reference_silences);
		actual_reference_silences = NULL;
	}
	if (actual_spectrogram != NULL)
	{
		free(actual_spectrogram);
		actual_spectrogram = NULL;
	}
	if (actual_train_words != NULL)
	{
		finalize_train_data(actual_train_words, actual_number_of_train_words);
		actual_train_words = NULL;
		actual_number_of_train_words = 0;
	}
	finalize_train_data_for_word(actual_train_silences);
	if (interesting_words != NULL)
	{
		finalize_interesting_words(interesting_words, number_of_interesting_words);
		interesting_words = NULL;
		number_of_interesting_words = 0;
	}
	remove(references_file_name);
}
