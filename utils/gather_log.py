# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_path', required=True, type=str)
  args = parser.parse_args()

  results = []
  with open(args.log_path, 'r') as f:
      lines = f.readlines()
  for line in lines:
      line = line.strip()
      if line.startswith('FinalResults'):
        predictor = line.split(' ')[1]
        prompt = line.split(' ')[3]
        score = line.split(' ')[5]
        results.append([prompt, predictor, score])
  
  print()
  print()
  print()
  headers = ['Prompt', 'Predictor', 'Score']
  print('| ' + ' | '.join(headers) + ' |')
  print('| ' + ' | '.join(['---'] * len(headers)) + ' |')
  for geneval, hps, score in results:
      print(f'| {geneval} | {hps} | {score} |')