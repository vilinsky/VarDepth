// Independent HTSlib-backed phased biallelic GT reader. No inference is done here.
#include <htslib/synced_bcf_reader.h>
#include <htslib/vcf.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: panel_io PANEL REGION EXCLUDED OUTPREFIX MAX_ALLELE START0 END0\n";
    return 2;
  }
  bcf_srs_t* reader = nullptr;
  int32_t* genotypes = nullptr;
  try {
    const int max_allele = std::stoi(argv[5]);
    const long start = std::stol(argv[6]), end = std::stol(argv[7]);
    std::unordered_set<std::string> excluded;
    std::ifstream ex(argv[3]);
    if (!ex) throw std::runtime_error("Cannot read excluded samples");
    for (std::string s; std::getline(ex, s);) excluded.insert(s);
    reader = bcf_sr_init();
    if (!reader) throw std::runtime_error("Cannot allocate HTS reader");
    if (bcf_sr_set_regions(reader, argv[2], 0) < 0 || !bcf_sr_add_reader(reader, argv[1]))
      throw std::runtime_error("Cannot open indexed panel/region");
    bcf_hdr_t* header = bcf_sr_get_header(reader, 0);
    const int samples = bcf_hdr_nsamples(header);
    std::vector<int> selected;
    std::string prefix(argv[4]);
    std::ofstream names(prefix + ".names"), sites(prefix + ".sites");
    std::ofstream matrix(prefix + ".gt", std::ios::binary);
    if (!names || !sites || !matrix) throw std::runtime_error("Cannot create output files");
    for (int i = 0; i < samples; ++i) {
      if (!excluded.count(header->samples[i])) {
        selected.push_back(i); names << header->samples[i] << '\n';
      }
    }
    if (selected.empty()) throw std::runtime_error("All panel samples excluded");
    std::vector<unsigned char> row(selected.size() * 2, 255);
    int capacity = 0;
    long rows = 0, unsupported = 0, unusable = 0;
    while (bcf_sr_next_line(reader)) {
      bcf1_t* record = bcf_sr_get_line(reader, 0);
      // Match pysam fetch: include REF-span overlaps; Python applies normalization.
      if (record->pos >= end || record->pos + record->rlen <= start) continue;
      bcf_unpack(record, BCF_UN_STR);
      bool valid = record->n_allele == 2;
      if (valid) for (int i = 0; i < 2; ++i) {
        std::string allele(record->d.allele[i]);
        if (allele.empty() || allele.size() > static_cast<size_t>(max_allele) ||
            allele.find_first_not_of("ACGT") != std::string::npos) valid = false;
      }
      if (!valid) { ++unsupported; continue; }
      const int count = bcf_get_genotypes(header, record, &genotypes, &capacity);
      const int stride = samples > 0 && count > 0 && count % samples == 0 ? count / samples : 0;
      for (size_t j = 0; j < selected.size(); ++j) {
        bool usable = stride >= 2;
        int a = -1, b = -1;
        if (usable) {
          const int32_t* gt = genotypes + selected[j] * stride;
          usable = gt[0] != bcf_int32_vector_end && gt[1] != bcf_int32_vector_end &&
                   !bcf_gt_is_missing(gt[0]) && !bcf_gt_is_missing(gt[1]);
          if (stride > 2 && gt[2] != bcf_int32_vector_end) usable = false;
          if (usable) {
            a = bcf_gt_allele(gt[0]); b = bcf_gt_allele(gt[1]);
            usable = a >= 0 && a <= 1 && b >= 0 && b <= 1 &&
                     (a == b || bcf_gt_is_phased(gt[1]));
          }
        }
        row[2*j] = usable ? a : 255;
        row[2*j+1] = usable ? b : 255;
        if (!usable) ++unusable;
      }
      sites << record->pos << '\t' << record->d.allele[0] << '\t' << record->d.allele[1] << '\n';
      matrix.write(reinterpret_cast<const char*>(row.data()), row.size());
      ++rows;
    }
    if (reader->errnum) throw std::runtime_error(bcf_sr_strerror(reader->errnum));
    names.close(); sites.close(); matrix.close();
    if (!names || !sites || !matrix) throw std::runtime_error("Output write failed");
    std::ofstream stats(prefix + ".json");
    stats << "{\"rows\":" << rows << ",\"columns\":" << row.size()
          << ",\"unsupported\":" << unsupported << ",\"unusable_sample_genotypes\":" << unusable << "}\n";
    stats.close();
    if (!stats) throw std::runtime_error("Metadata write failed");
    free(genotypes); bcf_sr_destroy(reader);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "panel_io: " << error.what() << '\n';
    free(genotypes); if (reader) bcf_sr_destroy(reader);
    return 1;
  }
}
