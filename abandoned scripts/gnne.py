
class GNNExplainer(nn.Module):
    def __init__(self, feature_reg_weight,
                 edge_reg_weight, **kwargs):

        super().__init__(**kwargs)
        self.feature_weight = feature_reg_weight
        self.edge_weight = edge_reg_weight
        # (n_edges, n_features)

    def entropy_regularization(self, soft_mask):
        n_edges, n_features = soft_mask.shape
        # clamp to avoid log(0)

        reg = 1e-8
        soft_mask = soft_mask.clamp(reg, 1 - reg)

        # edge entropy
        edge_mean_importances = torch.stack([
            torch.mean(soft_mask[i, :])
            for i in range(n_edges)
        ])
        edge_reg = self.edge_weight * torch.sum(edge_mean_importances * torch.log(edge_mean_importances))

        # feature entropy
        feature_entropies = torch.stack([
            -torch.sum(soft_mask[i, :] * torch.log(soft_mask[i, :]))
            for i in range(n_edges)
        ])
        feature_reg = self.feature_weight * torch.mean(feature_entropies)
        return edge_reg, feature_reg

    def fit(self, model, test_flows, epochs,
                 window, lr=0.01, loss_f=torch.nn.BCELoss()):

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        model.eval()
        for param in model.parameters():
              param.requires_grad = False

        for i, G in enumerate(yield_subgraphs(test_flows, window, linegraph=False)):
            num_features = len(test_flows.columns) - 3 # src dst Attack
            num_nodes = G.edge_index.max().item() + 1

            # initialize edge mask with requires_grad
            learned_mask = nn.Parameter(torch.randn(G.edge_attr.shape).to(device),
                                        requires_grad=True)
            optimizer = torch.optim.Adam([learned_mask], lr=lr)
            losses, edge_regularization, feature_regularization = [], [], []

            for epc in range(1, 1 + epochs):
                edge_weights = torch.sigmoid(learned_mask)
                optimizer.zero_grad()
                masked_edge_attr = G.edge_attr * edge_weights

                # F(G) — original predictions
                with torch.no_grad():
                    y_pred, _ = model.forward(G.edge_attr, G.edge_index,
                                              node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                    y_pred = torch.sigmoid(y_pred)

                # f(G_S) — masked predictions
                masked_y_pred, _ = model.forward(masked_edge_attr, G.edge_index,
                                                 node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                masked_y_pred = torch.sigmoid(masked_y_pred)

                loss = loss_f(masked_y_pred, y_pred)
                er, fr = self.entropy_regularization(edge_weights)
                total_loss = loss + er + fr

                total_loss.backward()
                optimizer.step()

                losses.append(loss.detach())
                edge_regularization.append(er.detach())
                feature_regularization.append(fr.detach())

                logger.info(
                    f'epoch: {epc} | '
                    f"av loss for mask: {np.mean(losses):.5f} | "
                    f"av edge regularizatino: {np.mean(edge_regularization):.5f} | "
                    f"av feature regularization: {np.mean(feature_regularization):.5f} | "
                    f"mask grad={learned_mask.grad.norm():.5f} "
                )

            losses_out = torch.stack(losses).cpu().numpy()
            edge_reg_out = torch.stack(edge_regularization).cpu().numpy()
            feature_reg_out = torch.stack(feature_regularization).cpu().numpy()

            yield (window, learned_mask, losses_out, edge_reg_out, feature_reg_out)

