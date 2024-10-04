<h2>Usage</h2>

<p>To run an experiment, use the following command:</p>

<pre><code>python run_exp.py -d lamp_5_dev -k 5 -f WF DPF -r contriever -ce 3
</code></pre>

<h3>Parameters:</h3>

<ul>
  <li><code>-d</code>, <code>--dataset</code>: <strong>Dataset name</strong>. 
    <ul>
      <li>Supported datasets include LaMP (4, 5, 7) and Amazon reviews.</li>
      <li>LaMP datasets follow the format <code>lamp_{dataset_num}_{data_split}</code> (e.g., <code>lamp_4_test</code>, <code>lamp_5_dev</code>) and use user-based splits by default.</li>
      <li>Amazon datasets follow the format <code>amazon_{category}_{year}</code> (e.g., <code>amazon_All_Beauty_2018</code>).</li>
      <li><strong>Default</strong>: <code>lamp_5_dev</code></li>
    </ul>
  </li>

  <li><code>-k</code>, <code>--top_k</code>: <strong>Number of retrieved documents for RAG</strong>. 
    <ul>
      <li>If <code>None</code> is passed, <code>k</code> is inferred by the length of the user profiles.</li>
      <li><strong>Default</strong>: <code>None</code></li>
    </ul>
  </li>

  <li><code>-f</code>, <code>--features</code>: <strong>Feature set</strong> (space-separated features). 
    <ul>
      <li><strong>Default</strong>: <code>None</code></li>
    </ul>
  </li>

  <li><code>-r</code>, <code>--retriever</code>: <strong>Retriever model</strong>.
    <ul>
      <li>Options include <code>&quot;contriever&quot;</code>, <code>&quot;dpr&quot;</code>, or any model available in <a href="https://www.sbert.net/">SentenceTransformers</a>.</li>
      <li><strong>Default</strong>: <code>contriever</code></li>
    </ul>
  </li>

  <li><code>-ce</code>, <code>--contrastive_examples</code>: <strong>Number of contrastive users</strong>. 
    <ul>
      <li>If <code>None</code>, contrastive examples method won’t be applied.</li>
      <li><strong>Default</strong>: <code>None</code></li>
    </ul>
  </li>

  <li><code>-ts</code>, <code>--two_step</code>: <strong>Boolean for deciding whether to use two-step generation</strong>. 
    <ul>
      <li><strong>Default</strong>: <code>False</code></li>
    </ul>
  </li>
</ul>

<h3>Evaluation:</h3>

<p>For evaluating a dataset:</p>

<pre><code>python eval.py -d dataset_name
</code></pre>
<p> This will create a csv file evaluating all the results found under the <em>preds</em> folder for the given dataset. It will create a csv file under <em>evals</em>. </p>
