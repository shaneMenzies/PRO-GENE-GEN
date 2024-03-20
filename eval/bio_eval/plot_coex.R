library(readr)
library(ggplot2)
library(magrittr)

data <- read_csv('bio_eval/outputs/coex_data_k5.csv')
real <- dplyr::filter(data, set == 'real')
data <- dplyr::filter(data, !set == 'real')


data_summary <- data.frame()

for(m in unique(data$model)){

  tmp <- dplyr::filter(data, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary <- rbind(data_summary, data.frame(
      mean_correct = mean(tmp_ep$correctly_rec),
      sd_correct = sd(tmp_ep$correctly_rec),
      mean_false = mean(tmp_ep$falsely_rec),
      sd_false = sd(tmp_ep$falsely_rec),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary$epsilon <- factor(data_summary$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary$seed <- 'k5'

data_summary1 <- data.frame(mean = data_summary$mean_correct,
                            sd = data_summary$sd_correct,
                            model = data_summary$model,
                            epsilon = data_summary$epsilon,
                            seed = data_summary$seed, 
                            reconstruction_type = 'correct')

data_summary2 <- data.frame(mean = data_summary$mean_false,
                            sd = data_summary$sd_false,
                            model = data_summary$model,
                            epsilon = data_summary$epsilon,
                            seed = data_summary$seed, 
                            reconstruction_type = 'false')

data_summary <- rbind(data_summary1, data_summary2)
data_summary$merged <- paste0(data_summary$model, '-', data_summary$reconstruction_type)


data_summary_correct <- dplyr::filter(data_summary, reconstruction_type == 'correct')
data_summary_false <- dplyr::filter(data_summary, reconstruction_type == 'false')
data_summary$model <- factor(data_summary$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))
data_summary_correct$model <- factor(data_summary_correct$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))
data_summary_false$model <- factor(data_summary_false$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))

g <- ggplot(data_summary, aes(x = epsilon, y = mean, fill = model, alpha = reconstruction_type))+
  facet_grid(.~model)+
  geom_bar(linewidth=2, position='dodge', stat="identity")+
  geom_line(data=data_summary_correct, aes(x = as.integer(epsilon)-0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
            linewidth=1)+
  geom_line(data=data_summary_false, aes(x = as.integer(epsilon)+0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
            linewidth=1)+
  geom_point(data=data_summary_correct, aes(x = as.integer(epsilon)-0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
            size=2)+
  geom_point(data=data_summary_false, aes(x = as.integer(epsilon)+0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
             size=2)+
  scale_alpha_manual(name = "", values = c(1, .3))+
  theme_bw()+
  geom_hline(yintercept = real$correctly_rec[1], linetype='dashed')+
  scale_fill_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  scale_color_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+
  ggtitle('Co-Expression Preservation')+
  ylab('Co-Expressions with r > 0.0')

g

print(real$correctly_rec[1])

data <- read_csv('bio_eval/outputs/coex_data_k1000.csv')
real <- dplyr::filter(data, set == 'real')
data <- dplyr::filter(data, !set == 'real')
print(real$correctly_rec[1])

data_summary2 <- data.frame()

for(m in unique(data$model)){

  tmp <- dplyr::filter(data, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary2 <- rbind(data_summary2, data.frame(
      mean_correct = mean(tmp_ep$correctly_rec),
      sd_correct = sd(tmp_ep$correctly_rec),
      mean_false = mean(tmp_ep$falsely_rec),
      sd_false = sd(tmp_ep$falsely_rec),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary2$epsilon <- factor(data_summary2$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary2$seed <- 'k1000'

data_summary1 <- data.frame(mean = data_summary2$mean_correct,
                            sd = data_summary2$sd_correct,
                            model = data_summary2$model,
                            epsilon = data_summary2$epsilon,
                            seed = data_summary2$seed, 
                            reconstruction_type = 'correct')

data_summary2 <- data.frame(mean = data_summary2$mean_false,
                            sd = data_summary2$sd_false,
                            model = data_summary2$model,
                            epsilon = data_summary2$epsilon,
                            seed = data_summary2$seed, 
                            reconstruction_type = 'false')

data_summary2 <- rbind(data_summary1, data_summary2)
data_summary2$merged <- paste0(data_summary2$model, '-', data_summary2$reconstruction_type)


data_summary_correct2 <- dplyr::filter(data_summary2, reconstruction_type == 'correct')
data_summary_false2 <- dplyr::filter(data_summary2, reconstruction_type == 'false')

data_summary2$model <- factor(data_summary2$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))
data_summary_correct2$model <- factor(data_summary_correct2$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))
data_summary_false2$model <- factor(data_summary_false2$model, levels = c('VAE', 'GAN', 'PGM', 'Privsyn', 'RONGauss'))

g2 <- ggplot()+
  
  geom_bar(data=data_summary2, aes(x = epsilon, y = mean, fill = model, alpha = reconstruction_type), 
           linewidth=2, position='dodge', stat="identity")+
  geom_line(data=data_summary_correct2, aes(x = as.integer(epsilon)-0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
            linewidth=1)+
  geom_line(data=data_summary_false2, aes(x = as.integer(epsilon)+0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
            linewidth=1)+
  geom_point(data=data_summary_correct2, aes(x = as.integer(epsilon)-0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
             size=2)+
  geom_point(data=data_summary_false2, aes(x = as.integer(epsilon)+0.25, y = mean, color = model, group = reconstruction_type, alpha = reconstruction_type),
             size=2)+
  facet_grid(cols = vars(model))+
  scale_alpha_manual(name = "", values = c(1, .3))+
  theme_bw()+
  geom_hline(yintercept = real$correctly_rec[1], linetype='dashed')+
  scale_fill_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  scale_color_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  ggtitle('')+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+
  ylab('Co-Expressions with r > 0.0')

g2



cp <- cowplot::plot_grid(g+theme(legend.position = 'none'), g2, nrow=1, rel_widths = c(1,1))
cp

png(file = paste0('bio_eval/outputs/coex-plot.png'), bg = "transparent", 
    width = 40, height = 10, units = "cm", res=300)
plot(cp)
dev.off()

data <- rbind(data_summary, data_summary2)
data$merged <- NULL

write_csv(data, 'bio_eval/outputs/coex-data-summarised.csv')
