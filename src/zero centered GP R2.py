"""
GP tanaka
GP
zero centered GP R1
zero centered GP R2

Gradient Penaltyの関数内のどこが違うのかを重点的に確認してください．
若干,train_step内も変わっています．


"""



discriminator_extra_steps = 5
latent_lr = 0.000002
gen_lr = 0.000002
disc_lr = 0.0000004

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to discriminator loss.
def discriminator_loss(true_data, recon_data):
    return -(tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data))

# Define the loss functions for the generator. 誤差のmaximaizeはtensorflowにはない．https://www.brainpad.co.jp/doors/contents/01_tech_2017-09-08-140000/
def generator_loss(true_data,recon_data): 
    return tf.math.reduce_mean(recon_data)

def zgenerator_loss(true_data,recon_data):
    return tf.math.reduce_mean(recon_data)



#WGAN-GPではAdamを，WGANではRMSpropをなんて話を目にした．https://techblog.cccmkhd.co.jp/entry/2022/04/26/074557 いや，原著論文読めよ俺．
zgenerator_optimizer = keras.optimizers.RMSprop(
    learning_rate = latent_lr
)

generator_optimizer = keras.optimizers.RMSprop(
    learning_rate = gen_lr
)

discriminator_optimizer = keras.optimizers.RMSprop(
    learning_rate = disc_lr
)



gen_array = []
latent_gen_array = []
Normalize_axis = 1
##########################################################

"""
GAN定義開始 Reference : https://www.tensorflow.org/guide/keras/custom_layers_and_models?hl=ja#putting_it_all_together_an_end-to-end_example

一口メモ

書き方をpytorchに寄せる．つまるところdefine by Runで学習を行うという事．fitにデータを入れるやつはdefine and run．
違いとしては
予め計算のシステムを作り上げて，そこにデータを流し込むやり方→define and run
データを生み出しながら正解と比較していくやり方→define by run


MD-GANは恐らく，後者に特化した作り方．論文の書き方を見るに，開発者はpyTorchで書いてるはず．
あちらはデフォルトでdefine by runしかできないが，Tensorflowはどちらもできる．（広く知れ渡ってるのはdefine and run）
Tensorflowにある．
Eager Executionというもの．これがdefine by runをtensorflowで可能にするインターフェース．
ver1.5まではsess.run()とかの宣言が必要だったとか．その後ver2.0を出してシンプルになったとか．

書き方はここと公式のリファレンスを見ながら
https://wshinya.hatenablog.com/entry/2019/10/18/144025
https://www.tensorflow.org/guide/function?hl=ja

@tf.functionをつけるとgraphモード               (define and run)
@tf.fuctionなし(コメントアウト)だとEagerモード   (define by run)

実際のところ，Eagerモードは計算が遅い（気がする．）ので，
二つの良いところ取りをするような訓練過程を構築しようと思う．

1. 初期値生成（論文に準拠）．Snz(一様乱数 [0,1])を初期値として，分布緩和を導入．7~10回の繰り返し生成で分布がおよそ一定になるらしい．

2. iteration回の勾配更新実行．(train_step関数呼び出し)

2-a. 前回の学習もしくは1.で生成した潜在変数を受け取る．
2-b. discriminatorの学習(WGANのリファレンスコードではgeneratorの5倍ほど学習回数を増やし，その代わり学習率を1/5にするみたいなことがあった)
2-c. generatorとlatent_generatorの学習（今は同時の学習にしているが，このやり方にする前はどちらでも精度や挙動が変わらなかった．だったらこっちのが処理早いし．）
2-d. 学習で生成した潜在変数をreturn(次の学習で使う)

3. d_lossとかの表示の奴をリセットする．じゃないとメモリを圧迫する．パソコンが逼迫する．

4. iterationが10000回（論文準拠）ごとにトラジェクトリを生成する（事が出来るモデルを保存する．）


"""


train_d_loss = tf.keras.metrics.Mean()
train_g_loss = tf.keras.metrics.Mean()
train_gz_loss = tf.keras.metrics.Mean()

#訓練過程の定義
###############################################################################
###############################################################################
###############################################################################
#GP_penaltyの実装
def GP(true_data,fake_data,batch_size):#Ishaaan Gulajani
    alpha = tf.random.uniform([batch_size,1,1],minval = 0.0,maxval = 1.0,dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data*alpha*diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated,training = True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm-1)**2)
    return gp

#---------------------- R2 regularization
def gradient_penaltyR2(fake_data,batch_size):
    fake_data = tf.cast(fake_data, tf.float64)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(fake_data)
        pred = discriminator(fake_data,training = True)
    
    grads = gp_tape.gradient(pred, [fake_data])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp

#---------------------- R1 regularization
def gradient_penaltyR1(true_data,batch_size):
    true_data = tf.cast(true_data, tf.float64)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(true_data)
        pred = discriminator(true_data,training = True)
    
    grads = gp_tape.gradient(pred, [true_data])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp


#---------------------- GP tanaka
def GP_tanaka(true_data,fake_data,batch_size):
    alpha = tf.random.uniform([batch_size,1,1],minval = 0.0,maxval = 1.0,dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data*alpha*diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated,training = True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp

def GP_tanaka_G(true_data,fake_data,batch_size):
    alpha = tf.random.uniform([batch_size,1,1],minval = 0.0,maxval = 1.0,dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data*alpha*diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated,training = True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp


#-----

def normalize_latent(latent):
    latent_mean = tf.math.reduce_mean(latent,axis=Normalize_axis,keepdims=True)
    std_mean = tf.math.reduce_std(latent,axis=Normalize_axis,keepdims=True)
    
    latent = (latent-latent_mean)/std_mean
    return latent

    
@tf.function()
def train_step(train_sample,latent_input):
    data = train_sample
    d_steps_add = discriminator_extra_steps
    gp_weight = 10
    ########################
    # discriminator学習開始 #
    ########################
    for l in range(d_steps_add):
        with tf.GradientTape() as tape :


            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #潜在変数生成
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent = latent_gen([latent_input,random_noise],training = False)
            latent = normalize_latent(latent)
            #潜在変数二つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent2 = latent_gen([latent,random_noise],training = False)
        
            latent2 = normalize_latent(latent2)
            #潜在変数三つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent3 = latent_gen([latent2,random_noise],training = False)
            latent3 = normalize_latent(latent3)

            #データ生成
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################
            reconstruction = generator([noise_inputer,latent],training = False)
            reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
            
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################
            reconstruction2 = generator([noise_inputer,latent2],training = False)
            reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])

            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################
            reconstruction3 = generator([noise_inputer,latent3],training = False)
            reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])
            ########################################################

            reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成終了~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #generated data の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            fake_logits_temp = discriminator(reconstruction_sample,training = True)
            fake_logits = fake_logits_temp
            
            #true data      の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            true_logits = discriminator(data,training = True)


            #損失関数の計算
            d_cost = discriminator_loss(true_logits,fake_logits)        # wasserstein loss
            d_cost = tf.cast(d_cost, tf.float64)

            gp =  gradient_penaltyR2(reconstruction_sample,batch_size)  # R2 regularization

            d_loss = d_cost + 1/2*gp*gp_weight                              # Wasserstein loss + GP
            

        d_gradient = tape.gradient(d_loss,discriminator.trainable_variables) #勾配の計算

        discriminator_optimizer.apply_gradients(
            zip(d_gradient,discriminator.trainable_variables)
        )

    ########################################
    # latent generator学習開始 #
    ########################################
    with tf.GradientTape() as tape :
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #潜在変数生成
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent = latent_gen([latent_input,random_noise],training = True)
    
        latent = normalize_latent(latent)
        #潜在変数二つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent2 = latent_gen([latent,random_noise],training = True)
        latent2 = normalize_latent(latent2)
        #潜在変数三つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent3 = latent_gen([latent2,random_noise],training = True)
        latent3 = normalize_latent(latent3)
        
        #データ生成
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction = generator([noise_inputer,latent],training = False)
        reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction2 = generator([noise_inputer,latent2],training = False)
        reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction3 = generator([noise_inputer,latent3],training = False)
        reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])
        ########################################################

        reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成終了~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #generated data の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fake_logits_temp = discriminator(reconstruction_sample,training = False)
        fake_logits = fake_logits_temp

        #true data      の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        true_logits = discriminator(data,training = False)

        #損失関数の計算
        gz_cost = zgenerator_loss(true_logits,fake_logits) #Wasserstein Loss
        gz_cost = tf.cast(gz_cost, tf.float64)

        gz_loss = gz_cost   #Wasserstein Loss のみ．（一般的には生成器にGPを与えないということ．つけてもいいかもしれない．）
        
    genz_gradient = tape.gradient(gz_loss,latent_gen.trainable_variables)

    zgenerator_optimizer.apply_gradients(
        zip(genz_gradient,latent_gen.trainable_variables)
    )


    ########################################
    # generator学習開始 #
    ########################################
    with tf.GradientTape() as tape :
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #潜在変数生成
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent1 = latent_gen([latent_input,random_noise],training = False)

        latent1 = normalize_latent(latent_input)
        #潜在変数二つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent2 = latent_gen([latent1,random_noise],training = False)

        latent2 = normalize_latent(latent2)
        #潜在変数三つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent3 = latent_gen([latent2,random_noise],training = False)

        latent3 = normalize_latent(latent3)
        
        #データ生成
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction = generator([noise_inputer,latent1],training = False)
        reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction2 = generator([noise_inputer,latent2],training = False)
        reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################
        reconstruction3 = generator([noise_inputer,latent3],training = False)
        reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])
        ########################################################

        reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~予測データ生成終了~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #generated data の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fake_logits_temp = discriminator(reconstruction_sample,training = False)
        fake_logits = fake_logits_temp

        #true data      の評価値生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        true_logits = discriminator(data,training = False)

        #損失関数の計算
        g_cost = generator_loss(true_logits,fake_logits) #Wasserstein Loss
        g_cost = tf.cast(g_cost, tf.float64)

        g_loss = g_cost   #Wasserstein Loss のみ．（一般的には生成器にGPを与えないということ．つけてもいいかもしれない．）
        
    gen_gradient = tape.gradient(g_loss,generator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gen_gradient,generator.trainable_variables)
    )

    train_d_loss.update_state(d_loss)
    train_g_loss.update_state(g_loss)
    train_gz_loss.update_state(gz_loss)

    return latent
